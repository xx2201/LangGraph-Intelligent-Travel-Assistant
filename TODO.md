# TODO

## 当前阶段：Phase 0 已完成

- [x] 建立 `src/` 项目结构
- [x] 提供一个可运行的最小图示例
- [x] 提供条件边示例
- [x] 提供基础测试
- [x] 提供 README 学习说明
- [x] 接入 LangSmith tracing
- [x] 接入 LangGraph Studio 本地开发模式

## Phase 1：改造成更适合 Studio 的输入方式

- [x] CLI 改成交互式输入模式，不再强依赖命令行附加参数
- [x] 增加更直观的输入字段，例如 `input`
- [x] 让 Studio 支持直接提交简单文本语义对应的 JSON
- [x] 减少手写完整状态对象的负担
- [x] 保持现有最小图逻辑仍然可运行

## Phase 2：进入 `MessagesState`

- [ ] 引入 `MessagesState`
- [ ] 把当前 `topic/background/notes` 风格过渡到消息流
- [ ] 在 README 中补充“普通状态 vs 消息状态”的对比
- [ ] 增加针对消息状态的测试

## Phase 3：接入真实模型

- [x] 加入真实 LLM 节点
- [x] 配置最小模型调用链路
- [x] 对比“确定性节点”和“模型节点”的差异
- [ ] 演示 `invoke()` 与 `stream()` 的区别

## Phase 4：工具与 Agent 化

- [ ] 加入一个最小工具调用节点
- [ ] 让图具备基本 agent 行为
- [ ] 在 Studio 中观察工具调用轨迹

## Phase 5：持久化与记忆

- [ ] 加入 `checkpointer` 和会话级 memory
- [ ] 演示 thread/session 的作用
- [ ] 对比“无状态执行”和“可恢复执行”

## Phase 6：进阶图结构

- [ ] 增加 `subgraph` 示例
- [ ] 增加错误处理与重试示例
- [ ] 增加 human-in-the-loop 示例
- [ ] 增加更复杂的分支与汇合结构

## 维护项

- [ ] 保持 README 与当前实现一致
- [ ] 每轮改动后更新本文件
- [ ] 每轮改动后在答复中给出中文总结，方便整理 Git 提交信息

## 学习方法提醒

- 先掌握图编排，再引入模型
- 先理解状态演化，再看 agent 复杂行为
- 每次只增加一个新概念，不要一次性把 memory、tool、agent、deployment 全混在一起
