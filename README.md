# macr_maddpg## v0.0.0测试git push的有效性## v1.0.0初始上传版本## v1.1.3   1. 动作解脱效果的判断机制：变为"出现冲突时就立刻停止"或者"时间到了300秒"；   2. 奖励函数      - 解脱失败，所有智能体获得-1.0的奖励/惩罚；       - 所有智能体奖励"取和"改为"最小值"；   3. 训练效果：统计增加了每回合的平均步数、回合解脱率以及回合平均奖励，删去了step_per_epi和与其相关的代码；   4. 其它：logs文件夹名称更改。## v1.2.4   1. 训练过程      - 恢复多次尝试解脱的机制，也即step_per_epi相关代码；      - 起始训练由步数限制改为回合数限制（1000个回合之后开始更新参数）；      - 增加A和c_type超参数，分别作为空间和时间的变量；   2. 测试过程：增加神经网络参数的可视化；   3. 训练效果：统计增加每百步尝试解脱的平均次数；## v1.3.1   1. 训练过程      - 再次去除多次尝试解脱的机制，增加var（噪声）的持续时间；      - 更改trained文件夹格式（按实验ID隔离logs、model和graph）；