#[cfg(test)]
mod tests {
    use crate::vm;

    #[test]
    fn vm_read() {
        let mem: Vec<u8>= vec![1, 2, 3, 4];
        let mut data = vm::Vm::read(&mem, 0, 3);
        assert_eq!(data, 0x010203 as u64);
        data = vm::Vm::read(&mem, 2, 2);
        assert_eq!(data, 0x0304 as u64);
    }

    #[test]
    fn vm_write() {
        let mut mem: Vec<u8>= vec![1, 2, 3, 4];
        vm::Vm::write(&mut mem, 7, 1, 2);
        assert_eq!(mem, [1, 0, 7, 4]);
        vm::Vm::write(&mut mem, 8, 1, 1);
        assert_eq!(mem, [1, 8, 7, 4]);
    }
}