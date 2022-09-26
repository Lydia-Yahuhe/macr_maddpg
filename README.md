# macr_maddpg# 版本更新记录## v0.0.0测试git push的有效性## v1.0.0初始上传版本## v1.1.3   1. 动作解脱效果的判断机制：变为"出现冲突时就立刻停止"或者"时间到了300秒"；   2. 奖励函数      - 解脱失败，所有智能体获得-1.0的奖励/惩罚；       - 所有智能体奖励"取和"改为"最小值"；   3. 训练效果：统计增加了每回合的平均步数、回合解脱率以及回合平均奖励，删去了step_per_epi和与其相关的代码；   4. 其它：logs文件夹名称更改。## v1.2.4   1. 训练过程      - 恢复多次尝试解脱的机制，也即step_per_epi相关代码；      - 起始训练由步数限制改为回合数限制（1000个回合之后开始更新参数）；      - 增加A和c_type超参数，分别作为空间和时间的变量；   2. 测试过程：增加神经网络参数的可视化；   3. 训练效果：统计增加每百步尝试解脱的平均次数；## v1.3.6   1. 训练过程      - 再次去除多次尝试解脱的机制，增加var（噪声）的持续时间；      - 更改trained文件夹格式（按实验ID隔离logs、model和graph）；      - states中每个智能体的state由原来的随机排列改成按position_in_bbox值排列；   2. 调试训练和测试，检查生成record文件的对错；   3. 调整训练和测试之间的联动性，使得测试时更加容易找到训练文件的所在地，以及训练和测试数据的文件融合修改（保存源数据，利用统一代码分析）；## v1.4.1   1. scenario,agentSet的代码重构，更加简洁和可读；    2. 状态中如果数量超过25，则分别从最小和最大方向剪除，保证数量为25；   3. 渲染优化（未完，缺少高度层表示）；## v2.0.3   1. 将经验回放池写入文件中；   2. 优化Memory和Model的更新数据格式(已放弃，v2.0.2之后改回原策略)；   3. 改变Critic的网络结构，RNN+FC改成FC+RNN+FC（已放弃，v2.0.3之后恢复）；   4. 改变Actor的网络结构，FC改成RNN+FC（v2.0.3）；# 实验## 第一组实验| 实验编号 | delta_T | delta_S | 解脱方式 | 是否共享策略网络 | 元学习 | 密度  | 是否完成 ||:----:|:-------:|:-------:|:----:|:--------:|:---:|:---:|:----:|| 1-1  |    0    |    1    | pair |    否     |  否  | 1/3 |      || 1-2  |    0    |    1    | pair |    否     |  否  | 1/2 |      || 1-3  |    0    |    1    | pair |    否     |  否  | 1/1 |      || 2-1  |    0    |    1    | conc |    是     |  否  | 1/3 |      || 2-2  |    0    |    1    | conc |    是     |  否  | 1/2 |      || 2-3  |    0    |    1    | conc |    是     |  否  | 1/1 |      ||  3   |    0    |    1    | conc |    是     |  是  | 1/1 |      |## 第二组实验| 实验编号 | delta_T | delta_S | 解脱方式 | 是否共享策略网络 | 元学习 | 密度  | 是否完成 ||:----:|:-------:|:-------:|:----:|:--------:|:---:|:---:|:----:|| 4-1  |    0    |    1    | conc |    是     |  是  |  1  |      || 4-2  |   60    |    1    | conc |    是     |  是  |  1  |      || 4-3  |   120   |    1    | conc |    是     |  是  |  1  |      || 5-1  |    0    |    4    | conc |    是     |  是  |  1  |      || 5-2  |   60    |    4    | conc |    是     |  是  |  1  |      || 5-3  |   120   |    4    | conc |    是     |  是  |  1  |      || 6-1  |    0    |    9    | conc |    是     |  是  |  1  |      || 6-2  |   60    |    9    | conc |    是     |  是  |  1  |      || 6-3  |   120   |    9    | conc |    是     |  是  |  1  |      |