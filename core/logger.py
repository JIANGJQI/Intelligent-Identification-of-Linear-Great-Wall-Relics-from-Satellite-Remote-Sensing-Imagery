"""
极简日志记录模块
功能：只将训练指标保存到txt文件，支持断点续训
"""

import os
from datetime import datetime


class TrainingLogger:
    """极简训练日志记录器"""

    def __init__(self, save_dir, experiment_name=None, resume_from=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 生成实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"train_{timestamp}"
        self.experiment_name = experiment_name

        # 日志文件路径
        self.log_path = os.path.join(save_dir, f"{experiment_name}.txt")
        
        # 记录是否是从断点恢复
        self.resume_from = resume_from

    def start(self):
        """开始记录"""
        # 如果是断点续训，以追加模式打开文件
        if self.resume_from and os.path.exists(self.log_path):
            self.log_file = open(self.log_path, 'a', encoding='utf-8')
            print(f"断点续训，日志追加至: {self.log_path}")
        else:
            # 新建文件，写入表头
            self.log_file = open(self.log_path, 'w', encoding='utf-8')
            self.log_file.write("# epoch\ttrain_loss\tbce\tdice\tlr\n")
            self.log_file.flush()
            print(f"新建日志: {self.log_path}")
        
        return self

    def log_epoch(self, epoch, train_loss, loss1u, loss2u, lr, **kwargs):
        """记录单个epoch的指标"""
        line = f"{epoch}\t{train_loss:.6f}\t{loss1u:.6f}\t{loss2u:.6f}\t{lr:.8f}\n"
        self.log_file.write(line)
        self.log_file.flush()
        
        # 每10个epoch打印一次进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.6f}")

    def close(self):
        """关闭日志文件"""
        if hasattr(self, 'log_file'):
            self.log_file.close()
            print(f"日志已保存: {self.log_path}")


# ========== 便捷函数 ==========
def create_logger(save_dir, experiment_name=None, resume_from=None):
    """创建日志记录器"""
    logger = TrainingLogger(save_dir, experiment_name, resume_from)
    return logger.start()