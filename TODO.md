# TODO

## 当前阶段：最小 LangGraph 学习骨架

- [x] 建立 `src/` 项目结构
- [x] 提供一个可运行的最小图示例
- [x] 提供条件边示例
- [x] 提供基础测试
- [x] 提供 README 学习说明
- [x] 提供中文迭代总结目录

## 下一阶段建议

- [ ] 引入 `MessagesState`，从普通状态过渡到对话状态
- [ ] 加入真实 LLM 节点
- [ ] 加入一个最小工具调用节点
- [ ] 演示 `stream()` 与执行过程观察
- [ ] 加入 `checkpointer` 和会话级 memory
- [ ] 增加 `subgraph` 示例
- [ ] 增加错误处理与重试示例
- [ ] 增加 human-in-the-loop 示例
- [ ] 接入 LangSmith tracing

## 学习方法提醒

- 先掌握图编排，再引入模型
- 先理解状态演化，再看 agent 复杂行为
- 每次只增加一个新概念，不要一次性把 memory、tool、agent、deployment 全混在一起

