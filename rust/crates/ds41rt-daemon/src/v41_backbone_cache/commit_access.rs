//! Static dispatch preserves direct single-GPU commits and scopes placed waves.
use super::*;

pub(crate) trait CacheWave<T> {
    fn wave_ref(&self) -> &T;
    fn on_device<R>(&self, operation: impl FnOnce(&T) -> Result<R>) -> Result<R>;
    fn on_device_mut<R>(&mut self, operation: impl FnOnce(&mut T) -> Result<R>) -> Result<R>;
}
impl<T> CacheWave<T> for DeviceOwner<'_, T> {
    fn wave_ref(&self) -> &T {
        self.get()
    }
    fn on_device<R>(&self, operation: impl FnOnce(&T) -> Result<R>) -> Result<R> {
        self.device.run(|| operation(self.get()))
    }
    fn on_device_mut<R>(&mut self, operation: impl FnOnce(&mut T) -> Result<R>) -> Result<R> {
        let device = self.device;
        device.run(|| operation(self.get_mut()))
    }
}
macro_rules! direct {
    ($wave:ident) => {
        impl<'w, 'a> CacheWave<$wave<'w, 'a>> for $wave<'w, 'a> {
            fn wave_ref(&self) -> &Self {
                self
            }
            fn on_device<R>(&self, operation: impl FnOnce(&Self) -> Result<R>) -> Result<R> {
                operation(self)
            }
            fn on_device_mut<R>(
                &mut self,
                operation: impl FnOnce(&mut Self) -> Result<R>,
            ) -> Result<R> {
                operation(self)
            }
        }
    };
}
direct!(WindowWave);
direct!(CompressorWave);
