// Standalone CPU row-delivery benchmark. No GPU work or production changes.
#include <liburing.h>
#include <aio.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <future>
#include <thread>

using Clock = std::chrono::steady_clock;
struct Read { off_t offset; size_t length, output; };
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string(message) + ": " + strerror(errno));
}
static void read_all(int fd, const Read& r, std::vector<char>& out) {
    size_t done = 0;
    while (done < r.length) {
        auto n = pread(fd, out.data() + r.output + done, r.length - done, r.offset + done);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "pread"); done += n;
    }
}
static double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b-a).count();
}
int main(int argc, char** argv) { try {
    if (argc != 10 && argc != 12) throw std::runtime_error(
        "usage: bench FILE WEIGHT_OFFSET SCALE_OFFSET ROW_COUNT TOKENS MODE CACHE SEED QDEPTH [LEAD_US gather|advisory]\n"
        "modes: mmap willneed populate pread uring uring-fixed uring-mmap aio posix-aio; cache: cold cold-global fresh warm");
    int fd = open(argv[1], O_RDONLY | O_CLOEXEC); require(fd >= 0, "open");
    int advice=posix_fadvise(fd,0,0,POSIX_FADV_RANDOM);
    if(advice) { errno=advice; require(false,"fadvise RANDOM"); }
    off_t weight = std::stoll(argv[2]), scale = std::stoll(argv[3]);
    uint64_t rows = std::stoull(argv[4]), tokens = std::stoull(argv[5]);
    std::string mode = argv[6], cache = argv[7];
    unsigned seed = std::stoul(argv[8]), depth = std::stoul(argv[9]);
    bool background=argc==12;
    long lead=background?std::stol(argv[10]):0;
    std::string delivery=background?argv[11]:"inline";
    if(lead<0 || lead>1000000 || (background && delivery!="gather" && delivery!="advisory"))
        throw std::runtime_error("invalid background delivery configuration");
    bool advisory=delivery=="advisory";
    if(advisory && mode!="willneed" && mode!="uring-mmap")
        throw std::runtime_error("advisory delivery requires willneed or uring-mmap");
    if (!rows || !tokens || tokens > 4096 || !depth || depth > 1024)
        throw std::runtime_error("invalid dimensions");
    if (cache != "cold" && cache != "cold-global" && cache != "fresh" && cache != "warm")
        throw std::runtime_error("invalid cache condition");
    bool mapped = mode == "mmap" || mode == "willneed" || mode == "populate";
    bool uring = mode == "uring" || mode == "uring-fixed" || mode == "uring-mmap";
    if (!mapped && !uring && mode != "pread" && mode != "aio" && mode != "posix-aio")
        throw std::runtime_error("invalid mode");
    struct stat st{}; require(fstat(fd, &st) == 0, "fstat");
    if (weight < 0 || scale < 0 || weight > st.st_size || scale > st.st_size ||
        rows > uint64_t(st.st_size-weight)/256 || rows > uint64_t(st.st_size-scale)/8)
        throw std::runtime_error("tensor outside file");
    std::mt19937_64 rng(seed);
    std::set<uint64_t> ids;
    // Synthetic uniform hashes, sorted as in the runtime's gather. No tokenization.
    for (size_t i=0; i<tokens*24; ++i) ids.insert(rng()%rows);
    std::vector<Read> reads;
    size_t output = 0;
    for (auto id : ids) {
        reads.push_back({weight+off_t(id*256),256,output}); output+=256;
        reads.push_back({scale+off_t(id*8),8,output}); output+=8;
    }
    std::vector<char> data(output, 0), reference(output, 0);
    long page = sysconf(_SC_PAGESIZE);
    std::set<off_t> pages;
    for (auto r : reads) for (off_t p=r.offset/page*page; p<r.offset+off_t(r.length); p+=page) pages.insert(p);
    io_uring ring{}; aio_context_t aio=0;
    if (uring) {
        int rc = io_uring_queue_init(depth,&ring,0);
        if (rc < 0) { errno=-rc; require(false,"io_uring_queue_init"); }
        if (mode == "uring-fixed") { iovec iov{data.data(),data.size()};
            rc=io_uring_register_buffers(&ring,&iov,1);
            if (rc < 0) { errno=-rc; require(false,"register buffers"); }
        }
    }
    if (mode=="aio") require(syscall(SYS_io_setup,depth,&aio)==0,"io_setup");
    if (uring) {
        size_t count=std::min<size_t>(depth,reads.size());
        for(size_t i=0;i<count;++i) {
            auto r=reads[i]; auto* sqe=io_uring_get_sqe(&ring);
            require(sqe!=nullptr,"warmup sqe");
            io_uring_prep_read(sqe,fd,data.data()+r.output,r.length,r.offset);
            sqe->flags |= IOSQE_ASYNC; sqe->user_data=i;
        }
        int submitted=io_uring_submit(&ring);
        require(submitted==int(count),"warmup submit");
        for(size_t i=0;i<count;++i) {
            io_uring_cqe* cqe=nullptr; int rc=io_uring_wait_cqe(&ring,&cqe);
            require(rc==0,"warmup wait");
            if(cqe->res!=int(reads.at(cqe->user_data).length)) throw std::runtime_error("warmup read failed");
            io_uring_cqe_seen(&ring,cqe);
        }
    }
    if(mode == "posix-aio") {
        // Prime the library worker before timing, as for io_uring's worker pool.
        aiocb b{}; auto r=reads.front(); b.aio_fildes=fd;
        b.aio_buf=data.data()+r.output; b.aio_nbytes=r.length; b.aio_offset=r.offset;
        b.aio_sigevent.sigev_notify=SIGEV_NONE;
        require(aio_read(&b)==0,"warmup aio_read");
        const aiocb* wait_for=&b; int error;
        while((error=aio_error(&b))==EINPROGRESS) {
            int rc=aio_suspend(&wait_for,1,nullptr);
            require(rc==0 || errno==EINTR,"warmup aio_suspend");
        }
        if(error) {errno=error;require(false,"warmup POSIX completion");}
        require(aio_return(&b)==ssize_t(r.length),"warmup POSIX read");
    }
    // Exact reference also primes the file cache before fresh/warm measurements.
    for (auto r : reads) read_all(fd,r,reference);
    if (cache == "cold") for (auto p : pages) {
        int rc = posix_fadvise(fd,p,page,POSIX_FADV_DONTNEED);
        if (rc) { errno=rc; require(false,"fadvise DONTNEED"); }
    }
    if (cache == "cold-global") {
        pid_t child=fork(); require(child>=0,"fork cache helper");
        if(child==0) {
            if(dup2(STDERR_FILENO,STDOUT_FILENO)<0) _exit(126);
            execl("/usr/local/libexec/ds41rt-bench/drop-page-cache", "drop-page-cache", nullptr);
            _exit(127);
        }
        int status=0; pid_t waited;
        do { waited=waitpid(child,&status,0); } while(waited<0 && errno==EINTR);
        require(waited==child,"wait cache helper");
        if(!WIFEXITED(status) || WEXITSTATUS(status)!=0) throw std::runtime_error("cache helper failed");
    }
    auto* map = static_cast<char*>(mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0));
    require(map != MAP_FAILED,"mmap");
    require(madvise(map,st.st_size,MADV_RANDOM)==0,"madvise RANDOM");
    size_t resident = 0;
    for (auto p : pages) { unsigned char state=0;
        require(mincore(map+p,page,&state)==0,"mincore"); resident += state&1;
    }
    if (cache == "warm" && (mapped || mode == "uring-mmap")) for (auto r : reads)
        memcpy(data.data()+r.output,map+r.offset,r.length);
    // Allocate all reusable submission/completion storage before timing.
    std::vector<iocb> blocks(depth); std::vector<iocb*> pointers(depth);
    std::vector<io_event> events(depth);
    std::vector<aiocb> posix_blocks(depth);
    Clock::time_point work_start,prepared,work_end;
    auto execute=[&]() {
    work_start=Clock::now();
    if (mode == "willneed" || mode == "populate") for (auto p : pages)
        require(madvise(map+p,page,mode=="willneed"?MADV_WILLNEED:MADV_POPULATE_READ)==0,"prefetch");
    prepared=Clock::now();
    if (mapped) {
        if(!advisory) for (auto r : reads) memcpy(data.data()+r.output,map+r.offset,r.length);
    }
    else if (mode=="pread") for (auto r : reads) read_all(fd,r,data);
    else for (size_t base=0;base<reads.size();base+=depth) {
        size_t count=std::min<size_t>(depth,reads.size()-base);
        if (uring) {
            for (size_t i=0;i<count;++i) { auto r=reads[base+i]; auto* sqe=io_uring_get_sqe(&ring);
                require(sqe!=nullptr,"get sqe");
                if (mode=="uring-fixed") io_uring_prep_read_fixed(sqe,fd,data.data()+r.output,r.length,r.offset,0);
                else io_uring_prep_read(sqe,fd,data.data()+r.output,r.length,r.offset);
                sqe->user_data=base+i;
            }
            size_t submitted=0;
            while (submitted<count) { int n=io_uring_submit(&ring);
                if(n<0){errno=-n;require(false,"uring submit");}
                require(n>0,"empty uring submit"); submitted+=n;
            }
            for(size_t i=0;i<count;++i){ io_uring_cqe* cqe=nullptr; int rc=io_uring_wait_cqe(&ring,&cqe);
                if(rc<0){errno=-rc;require(false,"uring wait");}
                if(cqe->res!=int(reads.at(cqe->user_data).length)) throw std::runtime_error("uring short/failed read");
                io_uring_cqe_seen(&ring,cqe);
            }
        } else if(mode == "aio") {
            for(size_t i=0;i<count;++i){ auto r=reads[base+i]; auto& b=blocks[i]; b={};
                b.aio_fildes=fd;b.aio_lio_opcode=IOCB_CMD_PREAD;b.aio_buf=uint64_t(data.data()+r.output);
                b.aio_nbytes=r.length;b.aio_offset=r.offset;b.aio_data=base+i;pointers[i]=&b;
            }
            size_t submitted=0;
            while(submitted<count){ long n=syscall(SYS_io_submit,aio,count-submitted,pointers.data()+submitted);
                require(n>0,"io_submit");submitted+=n;
            }
            size_t completed=0;
            while(completed<count){long n=syscall(SYS_io_getevents,aio,1,count-completed,events.data(),nullptr);
                require(n>0,"io_getevents");
                for(long i=0;i<n;++i) if(events[i].res!=int64_t(reads.at(events[i].data).length))
                    throw std::runtime_error("aio short/failed read");
                completed+=n;
            }
        } else {
            for(size_t i=0;i<count;++i) {
                auto r=reads[base+i]; auto& b=posix_blocks[i]; b={};
                b.aio_fildes=fd; b.aio_buf=data.data()+r.output;
                b.aio_nbytes=r.length; b.aio_offset=r.offset;
                b.aio_sigevent.sigev_notify=SIGEV_NONE;
                require(aio_read(&b)==0,"aio_read");
            }
            for(size_t i=0;i<count;++i) {
                auto& b=posix_blocks[i]; const aiocb* wait_for=&b;
                int error;
                while((error=aio_error(&b))==EINPROGRESS) {
                    int rc=aio_suspend(&wait_for,1,nullptr);
                    require(rc==0 || errno==EINTR,"aio_suspend");
                }
                if(error) {errno=error;require(false,"POSIX AIO completion");}
                if(aio_return(&b)!=ssize_t(reads[base+i].length))
                    throw std::runtime_error("POSIX AIO short read");
            }
        }
    }
    if(mode == "uring-mmap" && !advisory) {
        prepared=Clock::now();
        for(auto r : reads) memcpy(data.data()+r.output,map+r.offset,r.length);
    }
    work_end=Clock::now();
    };
    std::promise<void> ready,go;
    auto ready_future=ready.get_future(); auto go_future=go.get_future();
    std::exception_ptr failure; std::thread worker;
    if(background) {
        worker=std::thread([&] {
            ready.set_value(); go_future.wait();
            try {execute();} catch(...) {failure=std::current_exception();}
        });
        ready_future.wait();
    }
    rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
    auto start=Clock::now(); auto consume=start;
    if(background) {
        go.set_value();
        std::this_thread::sleep_until(start+std::chrono::microseconds(lead));
        consume=Clock::now(); worker.join();
        if(failure) std::rethrow_exception(failure);
        if(advisory) for(auto r : reads) memcpy(data.data()+r.output,map+r.offset,r.length);
    } else execute();
    auto end=Clock::now();getrusage(RUSAGE_SELF,&after);
    if(data!=reference) throw std::runtime_error("row bytes differ");
    std::cout << "{\"mode\":\""<<mode<<"\",\"cache\":\""<<cache<<"\",\"tokens\":"<<tokens
        <<",\"seed\":"<<seed<<",\"depth\":"<<depth<<",\"unique_rows\":"<<ids.size()
        <<",\"pages\":"<<pages.size()<<",\"resident_before\":"<<resident
        <<",\"prefetch_us\":"<<us(work_start,prepared)<<",\"gather_us\":"<<us(prepared,work_end)
        <<",\"total_us\":"<<us(start,end)<<",\"minor_faults\":"<<after.ru_minflt-before.ru_minflt
        <<",\"major_faults\":"<<after.ru_majflt-before.ru_majflt
        <<",\"input_blocks\":"<<after.ru_inblock-before.ru_inblock<<",\"bytes\":"<<output
        <<",\"delivery\":\""<<delivery<<"\",\"lead_us\":"<<lead
        <<",\"actual_lead_us\":"<<us(start,consume)<<",\"residual_us\":"<<us(consume,end)
        <<",\"worker_done_us\":"<<us(start,work_end)<<",\"exact\":true}\n";
    if(uring) io_uring_queue_exit(&ring);
    if(aio) syscall(SYS_io_destroy,aio);
    munmap(map,st.st_size);close(fd);
    return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; } }
