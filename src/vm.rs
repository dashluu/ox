use std::fs;

#[derive(FromPrimitive)]
enum Opcode {
    STORE_LOCAL, STORE_ARG,
    LOAD_VAL, LOAD_LOCAL, LOAD_ARR, LOAD_LITERAL, LOAD_SCOPE,
    F32_PLUS, F32_MINUS, F32_ADD, F32_SUB, F32_MUL, F32_DIV,
    I32_PLUS, I32_MINUS, I32_ADD, I32_SUB, I32_MUL, I32_DIV, MOD,
    SHL, ART_SHR, LOG_SHR,
    BW_NOT, BW_AND, BW_OR, BW_XOR,
    LOG_NOT, LOG_AND, LOG_OR,
    F32_GREATER, F32_LESS, F32_GEQ, F32_LEQ,
    I32_GREATER, I32_LESS, I32_GEQ, I32_LEQ,
    NEQ, EQ, ASSIGN,
    I32_TO_F32, F32_TO_I32,
    PUSH, PEEK,
    JMP_OVER_FUN, JMP_OUT_LOOP, JMP_TO_LOOP, JMP_LOOP_COND_FALSE, JMP_IF_COND_FALSE, JMP_OUT_IF_ELSE,
    PRINT_I32, PRINT_F32, PRINT_BOOL,
    RET, CALL, EXIT
}

#[derive(Copy, Clone)]
struct Variable {
    reference: usize,
    size: usize,
}

pub struct Vm {
    const_pool: Vec<u8>,
    bytecode: Vec<u64>,
    call_stack: Vec<u8>,
    eval_stack: Vec<u8>,
    var_stack: Vec<Variable>,
    scope_stack: Vec<usize>,
    call_stack_ptr: usize,
    eval_stack_ptr: usize,
    frame_ptr: usize,
    instr_ptr: usize,
    var_stack_ptr: usize,
    scope_stack_ptr: usize,
}

impl Vm {
    const MODE: usize = 4;
    const CALL_STACK_SIZE: usize = 128;
    const EVAL_STACK_SIZE: usize = 128;
    const VAR_STACK_SIZE: usize = 128;
    const SCOPE_STACK_SIZE: usize = 128;
    const OPCODE_SIZE: usize = 1;
    const INSTR_SIZE: usize = 8;
    const OPERAND_SIZE: usize = Vm::INSTR_SIZE - Vm::OPCODE_SIZE;
    const OPCODE_MASK: u64 = !(u64::MAX << (8 * Vm::OPCODE_SIZE));
    const OPERAND_MASK: u64 = !(u64::MAX << (8 * Vm::OPERAND_SIZE));

    pub fn new(bytecode_fn: &str) -> Vm {
        let buff = fs::read(bytecode_fn).unwrap();
        let const_pool_size = Vm::read(&buff, 0, Vm::MODE) as usize;
        let bytecode_size = (buff.len() - const_pool_size) / Vm::INSTR_SIZE;

        let mut vm_inst = Vm {
            const_pool: vec![0; const_pool_size],
            bytecode: vec![0; bytecode_size],
            call_stack: vec![0; Vm::CALL_STACK_SIZE],
            eval_stack: vec![0; Vm::EVAL_STACK_SIZE],
            var_stack: vec![Variable {reference: 0, size: 0}; Vm::VAR_STACK_SIZE],
            scope_stack: vec![0; Vm::SCOPE_STACK_SIZE],
            call_stack_ptr: 0,
            eval_stack_ptr: 0,
            frame_ptr: 0,
            instr_ptr: 0,
            var_stack_ptr: 0,
            scope_stack_ptr: 0,
        };

        // Initialize constant pool
        for i in 0..const_pool_size {
            vm_inst.const_pool[i] = buff[i];
        }

        // Initialize bytecode
        let mut addr: usize;
        for i in 0..bytecode_size {
            addr = (i * Vm::INSTR_SIZE) + const_pool_size;
            vm_inst.bytecode[i] = Vm::read(&buff, addr, Vm::INSTR_SIZE);
        }

        // Initialize scope
        vm_inst.push_scope_stack(0);

        return vm_inst;
    }

    // Note: this follows big endian convention
    pub fn read(mem: &Vec<u8>, addr: usize, size: usize) -> u64 {
        let mut result: u64 = 0;
        for i in 0..size {
            result = (result << 8) | (mem[addr + i] as u64);
        }
        return result;
    }

    // Note: this follows big endian convention
    pub fn write(mem: &mut Vec<u8>, data: u64, addr: usize, size: usize) {
        for i in 0..size {
            mem[addr + (size - 1 - i)] = ((data >> (8 * i)) & 0xff) as u8;
        }
    }

    fn read_const_pool(&self, addr: usize, size: usize) -> u64 {
        return Vm::read(&self.const_pool, addr, size);
    }

    fn read_call_stack(&self, addr: usize, size: usize) -> u64 {
        return Vm::read(&self.call_stack, addr, size);
    }

    fn write_call_stack(&mut self, data: u64, addr: usize, size: usize) {
        Vm::write(&mut self.call_stack, data, addr, size);
    }

    fn push_call_stack(&mut self, data: u64, size: usize) {
        self.write_call_stack(data, self.call_stack_ptr, size);
        self.call_stack_ptr += size;
    }

    fn pop_call_stack(&mut self, size: usize) -> u64 {
        let data = self.peek_call_stack(size);
        self.call_stack_ptr -= size;
        return data;
    }

    fn peek_call_stack(&self, size: usize) -> u64 {
        return self.read_call_stack(self.call_stack_ptr - size, size);
    }

    fn push_eval_stack(&mut self, data: u64, size: usize) {
        Vm::write(&mut self.eval_stack, data, self.eval_stack_ptr, size);
        self.eval_stack_ptr += size;
    }

    fn pop_eval_stack(&mut self, size: usize) -> u64 {
        let data = self.peek_eval_stack(size);
        self.eval_stack_ptr -= size;
        return data;
    }

    fn peek_eval_stack(&self, size: usize) -> u64 {
        return Vm::read(&self.eval_stack, self.eval_stack_ptr - size, size);
    }

    fn push_var_stack(&mut self, variable: Variable) {
        self.var_stack[self.var_stack_ptr] = variable;
        self.var_stack_ptr += 1;
    }

    fn push_scope_stack(&mut self, scope: usize) {
        self.scope_stack[self.scope_stack_ptr] = scope;
        self.scope_stack_ptr += 1;
    }

    fn pop_scope_stack(&mut self) -> usize {
        let scope = self.scope_stack[self.scope_stack_ptr - 1];
        self.scope_stack_ptr -= 1;
        return scope;
    }

    fn get_opcode(instr: u64) -> u64 {
        // Be careful when converting enums to u64s
        return (instr >> (8 * Vm::OPERAND_SIZE)) & Vm::OPCODE_MASK;
    }

    fn get_operand(instr: u64) -> u64 {
        return instr & Vm::OPERAND_MASK;
    }

    fn print_u8_mem(label: &str, mem: &Vec<u8>, ptr: usize) {
        let mut output = label.to_string();
        let mut data_str: String;
        let mut data: u64;
        for i in 0..ptr {
            data = Vm::read(mem, i, 1);
            data_str = format!("{:x}", data) + " ";
            output.push_str(&data_str);
        }
        println!("{output}");
    }

    fn print_usize_mem(label: &str, mem: &Vec<usize>, ptr: usize) {
        let mut output = label.to_string();
        let mut data_str: String;
        let mut data: usize;
        for i in 0..ptr {
            data = mem[i];
            data_str = format!("{:x}", data) + " ";
            output.push_str(&data_str);
        }
        println!("{output}");
    }

    fn print_u64_mem(label: &str, mem: &Vec<u64>, ptr: usize) {
        let mut output = label.to_string();
        let mut data_str: String;
        let mut data: u64;
        for i in 0..ptr {
            data = mem[i];
            data_str = format!("{:x}", data) + " ";
            output.push_str(&data_str);
        }
        println!("{output}");
    }

    pub fn print_const_pool(&self) {
        Vm::print_u8_mem("Constant pool: ", &self.const_pool, self.const_pool.len());
    }

    pub fn print_instr_file(&self) {
        Vm::print_u64_mem("Instruction file: ", &self.bytecode, self.bytecode.len());
    }

    pub fn print_call_stack(&self) {
        Vm::print_u8_mem("Call stack: ", &self.call_stack, self.call_stack_ptr);
        let call_stack_ptr_str = format!("Call stack pointer: {}", self.call_stack_ptr);
        println!("{call_stack_ptr_str}");
    }

    pub fn print_eval_stack(&self) {
        Vm::print_u8_mem("Evaluation stack: ", &self.eval_stack, self.eval_stack_ptr);
        let eval_stack_ptr_str = format!("Evaluation stack pointer: {}", self.eval_stack_ptr);
        println!("{eval_stack_ptr_str}");
    }

    pub fn print_var_stack(&self) {
        let mut var_stack_str = "Variable stack: ".to_string();
        let mut data_str: String;
        let mut data: Variable;
        for i in 0..self.var_stack_ptr {
            data = self.var_stack[i];
            data_str = format!("(ref={}, size={})", data.reference, data.size) + " ";
            var_stack_str.push_str(&data_str);
        }
        println!("{var_stack_str}");
        let var_stack_ptr_str = format!("Variable stack pointer: {}", self.var_stack_ptr);
        println!("{var_stack_ptr_str}");
    }

    pub fn print_scope_stack(&self) {
        Vm::print_usize_mem("Scope stack: ", &self.scope_stack, self.scope_stack_ptr);
        let scope_ptr_str = format!("Scope pointer: {}", self.scope_stack_ptr);
        println!("{scope_ptr_str}");
    }

    fn print_instr(&mut self, opcode: Option<Opcode>) {
        let i32_operand: i32;
        let f32_operand: f32;
        let bool_operand: bool;

        match opcode {
            Some(Opcode::PRINT_I32) => {
                i32_operand = self.pop_call_stack(4) as i32;
                println!("{i32_operand}");
            }
            Some(Opcode::PRINT_F32) => {
                f32_operand = f32::from_bits(self.pop_call_stack(4) as u32);
                println!("{f32_operand}")
            }
            _ => {
                bool_operand = self.pop_call_stack(1) != 0;
                println!("{bool_operand}")
            }
        }
    }

    fn bin_op_instr(&mut self, opcode: Option<Opcode>) {
        let i32_operand1: i32;
        let i32_operand2: i32;
        let i32_result: i32;
        let f32_operand1: f32;
        let f32_operand2: f32;
        let f32_result: f32;

        match opcode {
            Some(Opcode::I32_ADD) => {
                i32_operand1 = self.pop_eval_stack(4) as i32;
                i32_operand2 = self.pop_eval_stack(4) as i32;
                i32_result = i32_operand1 + i32_operand2;
                self.push_eval_stack(i32_result as u64, 4);
            }
            _ => {
                f32_operand1 = f32::from_bits(self.pop_eval_stack(4) as u32);
                f32_operand2 = f32::from_bits(self.pop_eval_stack(4) as u32);
                f32_result = f32_operand1 + f32_operand2;
                self.push_eval_stack( f32_result.to_bits() as u64, 4);
            }
        }
    }

    pub fn run(&mut self) {
        let mut instr: u64;
        let mut opcode_num: u64;
        let mut opcode: Option<Opcode>;
        let mut operand: u64;
        let mut ret_val: u64 = 0;
        let mut data: u64;
        
        let mut next_instr_ptr: usize;
        let mut variable: Variable;
        let mut reference: usize;
        let mut num_args: usize = 0;
        let mut size: usize;
        let mut scope: usize;
        let mut base: usize;
        let mut offset: isize;
        let mut end = false;
        let mut instr_line: usize = 0;

        while !end {
            instr = self.bytecode[self.instr_ptr];
            // Get the first byte as the opcode
            opcode_num = Vm::get_opcode(instr);
            opcode = num::FromPrimitive::from_u64(opcode_num);
            operand = Vm::get_operand(instr);

            match opcode {
                Some(Opcode::LOAD_SCOPE) => {
                    scope = operand as usize;
                    self.push_call_stack(scope as u64, Vm::MODE);
                    self.instr_ptr += 1;
                }
                Some(Opcode::STORE_LOCAL) => {
                    size = operand as usize;
                    reference = self.call_stack_ptr - size;
                    self.push_var_stack(Variable { reference, size });
                    self.instr_ptr += 1;
                }
                Some(Opcode::STORE_ARG) => {
                    size = operand as usize;
                    reference = self.call_stack_ptr - size;
                    self.push_var_stack(Variable { reference, size });
                    num_args += 1;
                    self.instr_ptr += 1;
                }
                Some(Opcode::LOAD_VAL) => {
                    size = operand as usize;
                    reference = self.pop_call_stack(Vm::MODE) as usize;
                    data = self.read_call_stack(reference, size);
                    self.push_call_stack(data, size);
                    self.instr_ptr += 1;
                }
                Some(Opcode::LOAD_LOCAL) => {
                    if ((operand >> (8 * Vm::OPERAND_SIZE - 1)) & 1) == 1 {
                        // If the msb of the operand is 1, it is negative so sign-extend it
                        offset = (operand | (Vm::OPCODE_MASK << (8 * Vm::OPERAND_SIZE))) as isize;
                    } else {
                        offset = operand as isize;
                    }
                    // The variable scope has already been pushed onto the stack
                    scope = self.pop_call_stack(Vm::MODE) as usize;
                    base = self.scope_stack[scope];
                    reference = (base as isize + offset) as usize;
                    variable = self.var_stack[reference];
                    self.push_call_stack(variable.reference as u64, Vm::MODE);
                    self.instr_ptr += 1;
                }
                Some(Opcode::LOAD_LITERAL) => {
                    reference = operand as usize;
                    size = self.read_const_pool(reference, Vm::MODE) as usize;
                    // Each literal is preceded by its size of 4 bytes
                    data = self.read_const_pool(reference + Vm::MODE, size);
                    self.push_call_stack(data, size);
                    self.instr_ptr += 1;
                }
                Some(Opcode::LOAD_ARR) => {
                    size = operand as usize;
                    offset = self.pop_call_stack(Vm::MODE) as isize;
                    base = self.pop_call_stack(Vm::MODE) as usize;
                    reference = (base as isize + size as isize * offset) as usize;
                    self.push_call_stack(reference as u64, Vm::MODE);
                    self.instr_ptr += 1
                }
                Some(Opcode::I32_TO_F32) => {
                    size = operand as usize;
                    data = self.pop_call_stack(size);
                    data = (data as i32 as f32).to_bits() as u64;
                    self.push_call_stack(data, 4);
                    self.instr_ptr += 1;
                }
                Some(Opcode::F32_TO_I32) => {
                    size = operand as usize;
                    data = self.pop_call_stack(size);
                    data = f32::from_bits(data as u32) as i32 as u64;
                    self.push_call_stack(data, 4);
                    self.instr_ptr += 1;
                }
                Some(Opcode::PUSH) => {
                    size = operand as usize;
                    data = self.peek_call_stack(size);
                    self.push_eval_stack(data, size);
                    self.instr_ptr += 1;
                }
                Some(Opcode:: PEEK) => {
                    size = operand as usize;
                    data = self.peek_eval_stack(size);
                    self.push_call_stack(data, size);
                    self.instr_ptr += 1;
                }
                Some(
                    Opcode::JMP_IF_COND_FALSE
                    | Opcode::JMP_LOOP_COND_FALSE
                    | Opcode::JMP_OUT_IF_ELSE
                    | Opcode::JMP_OUT_LOOP
                    | Opcode::JMP_OVER_FUN
                    | Opcode::JMP_TO_LOOP,
                ) => {
                    self.instr_ptr = operand as usize;
                }
                Some(Opcode::CALL) => {
                    // Push the next instruction onto the stack to resume the program later
                    next_instr_ptr = self.instr_ptr + 1;
                    self.push_call_stack(next_instr_ptr as u64, Vm::MODE);
                    // Push the current frame pointer content onto the stack to recover states later
                    self.push_call_stack(self.frame_ptr as u64, Vm::MODE);
                    // Update the frame pointer
                    self.frame_ptr = self.call_stack_ptr;
                    // New scope points to the top of the variable stack
                    self.push_scope_stack(self.var_stack_ptr);
                    // Jump to the function
                    self.instr_ptr = operand as usize;
                }
                Some(Opcode::RET) => {
                    size = operand as usize;
                    if size > 0 {
                        // Pop the return value off the stack if size > 0
                        ret_val = self.pop_call_stack(size);
                    }
                    // Pop the current scope off the scope stack
                    self.var_stack_ptr = self.pop_scope_stack() - num_args;
                    // Clear the number of arguments after returning
                    num_args = 0;
                    // Adjust the stack pointer
                    self.call_stack_ptr = self.frame_ptr;
                    // Pop the last frame pointer off the stack
                    self.frame_ptr = self.pop_call_stack(Vm::MODE) as usize;
                    // Pop the instruction to be resumed
                    self.instr_ptr = self.pop_call_stack(Vm::MODE) as usize;
                    if size > 0 {
                        // Push the return value onto the stack if size > 0
                        self.push_call_stack(ret_val, size);
                    }
                }
                Some(Opcode::EXIT) => {
                    end = true;
                }
                Some(Opcode::PRINT_I32 | Opcode::PRINT_F32 | Opcode::PRINT_BOOL) => {
                    self.print_instr(opcode);
                    self.instr_ptr += 1;
                }
                _ => {
                    self.bin_op_instr(opcode);
                    self.instr_ptr += 1;
                }
            }

            // println!("{instr_line} {opcode_num}");
            // self.print_call_stack();
            // self.print_eval_stack();
            // self.print_var_stack();
            // self.print_scope_stack();
            // instr_line += 1;
        }
    }
}
