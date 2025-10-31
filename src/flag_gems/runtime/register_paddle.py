class RegisterPaddle:
    def __init__(self, config, user_unused_ops_list=None):
        """
        Paddle算子注册器
        
        Args:
            config: 算子配置列表，格式为 [(op_name, custom_op), ...]
                   其中op_name是算子名称，custom_op是自定义算子函数
            user_unused_ops_list: 用户指定不使用的算子列表
        """
        self.all_ops = []
        self.unused_ops = user_unused_ops_list or []
        self.config = config
        self.original_ops = {}  # 保存原始算子
        self.config_filter()
        self.for_each()

    def config_filter(self):
        """过滤掉不需要注册的算子"""
        self.config = [
            item for item in self.config 
            if item[1].__name__ not in self.unused_ops
        ]

    def register_impl(self, op_name, custom_op):
        """
        注册算子实现
        
        Args:
            op_name: 算子名称（支持嵌套属性，如 'nn.functional.softmax'）
            custom_op: 自定义算子函数
        """
        import paddle
        # 处理嵌套属性（如 'nn.functional.softmax'）
        if '.' in op_name:
            # 分割路径
            parts = op_name.split('.')
            module_path = parts[:-1]  # ['nn', 'functional']
            attr_name = parts[-1]     # 'softmax'
            
            # 获取最终模块
            
            module = paddle
            for part in module_path:
                module = getattr(module, part)
            
            # 保存原始算子
            original_op = getattr(module, attr_name)
            self.original_ops[op_name] = (module, attr_name, original_op)
            
            # 替换为自定义算子
            setattr(module, attr_name, custom_op)
        else:
            # 直接属性（如 'tanh'）
            original_op = getattr(paddle, op_name)
            self.original_ops[op_name] = (paddle, op_name, original_op)
            
            # 替换为自定义算子
            setattr(paddle, op_name, custom_op)
        
        self.all_ops.append(op_name)

    def for_each(self):
        """遍历配置并注册所有算子"""
        try:
            for item in self.config:
                if len(item) == 2:
                    op_name, custom_op = item
                    self.register_impl(op_name, custom_op)
                else:
                    print(f"Invalid config format: {item}. Expected (op_name, custom_op)")
        except Exception as e:
            print(f"Error registering paddle ops: {e}")

    def restore_all(self):
        """恢复所有原始算子"""
        for op_name, saved_info in self.original_ops.items():
            module, attr_name, original_op = saved_info
            setattr(module, attr_name, original_op)

    def restore_op(self, op_name):
        """恢复指定的原始算子"""
        if op_name in self.original_ops:
            module, attr_name, original_op = self.original_ops[op_name]
            setattr(module, attr_name, original_op)
            print(f"Restored paddle op: {op_name}")
        else:
            print(f"Original op not found: {op_name}")

    def get_all_ops(self):
        """获取所有已注册的算子"""
        return self.all_ops

    def get_unused_ops(self):
        """获取未使用的算子"""
        return self.unused_ops