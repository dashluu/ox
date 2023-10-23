extern crate num;
#[macro_use]
extern crate num_derive;
mod vm;
mod test;

fn main() {
    let bytecode_fn = "code.tinycode";
    let mut vm_inst = vm::Vm::new(bytecode_fn);
    vm_inst.run();
}


