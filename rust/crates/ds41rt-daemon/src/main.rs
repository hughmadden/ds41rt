#[tokio::main]
async fn main() -> anyhow::Result<()> {
    ds41rt_daemon::run_cli().await
}
