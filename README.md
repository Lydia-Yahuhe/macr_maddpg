# macr_maddpg## v0.0.0测试git push的有效性## v1.0.0初始上传版本## v1.1.01、动作解脱效果的判断机制：变为"出现冲突时就立刻停止"或者"时间到了300秒"；2、奖励函数：解脱失败，所有智能体获得-1.0的奖励/惩罚；3、奖励函数：所有智能体奖励"取和"改为"最小值"；4、训练效果：统计增加了每回合的平均步数、回合解脱率以及回合平均奖励，删去了step_per_epi和与其相关的代码；5、其它：logs文件夹名称更改。