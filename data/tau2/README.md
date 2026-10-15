# tau retail 数据

来源：[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)。

固定版本：`5bfa7e37b36656b37dc6d022156be6563c1007f3`；MIT许可证见下载目录。原始数据保留英文，不自行翻译以免改变官方评测语义。

原始业务文件在`third_party/tau2-bench/data/tau2/domains/retail/`：

- `tasks.json`：114个任务场景及评测标准。
- `db.json`：模拟用户、订单、商品等初始状态。
- `policy.md`：客服必须遵守的业务规则。
- `split_tasks.json`：官方74个训练任务与40个测试任务。

本目录自动生成`upstream-manifest.json`（源码/数据哈希）与`splits.json`（本项目59训练、15开发、40测试ID）。划分只从官方训练部分留出开发集，并分组完全相同的用户场景；不保证不同表达模板或业务实体绝对独立。

这不是生产聊天记录，也不是预制SFT集。SFT需要在训练任务上采集教师多轮交互，再筛选到`data/sft/`。每个任务可采多次，但不能将轨迹数量描述为独立场景数量。

恢复下载：从项目根目录运行`PYTHONPATH=src python -m mini_customer_agent.retail.data`；首次运行`retail-agent inspect`创建固定划分。生成文件不提交Git，恢复过程可重复执行并校验版本。
