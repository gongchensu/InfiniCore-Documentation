# `CrossEntropyLoss`

`CrossEntropyLoss`, 即**交叉熵损失**算子，计算输入 logits 和目标类别之间的交叉熵损失，常用于分类任务的损失函数。

对于输入 logits 张量 $x$ 和目标类别张量 $t$，交叉熵损失的计算公式为：

$$ \text{loss} = -\frac{1}{N} \sum_{i=1}^{N} \log\left(\frac{\exp(x_{i,t_i})}{\sum_{j=1}^{C} \exp(x_{i,j})}\right) $$

其中：
- $N$ 是样本数量
- $C$ 是类别数量
- $x_{i,j}$ 是第 $i$ 个样本的第 $j$ 个类别的 logit 值
- $t_i$ 是第 $i$ 个样本的目标类别索引

## 接口

### 计算

```c
infiniStatus_t infiniopCrossEntropyLoss(
    infiniopCrossEntropyLossDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *loss,
    const void *logits,
    const void *target,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 已使用 `infiniopCreateCrossEntropyLossDescriptor()` 初始化的算子描述符；
- `workspace`: 指向算子计算所需的额外工作空间；
- `workspace_size`: `workspace` 的大小，单位：字节；
- `loss`: 输出损失值张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `logits`: 输入 logits 张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `target`: 目标类别索引张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`: 计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateCrossEntropyLossDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyLossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t loss_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t target_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: `infiniopCrossEntropyLossDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `loss_desc` - { dT | shape }:
  输出损失值张量描述。形状取决于输入 logits 的形状，通常为标量或与 logits 的前 N-1 维相同。
- `logits_desc` - { dT | (..., C) | (...) }:
  输入 logits 张量描述。最后一个维度为类别数 `C`，前面的维度为批次和空间维度。
- `target_desc` - { Int32 | (...,) | (...) }:
  目标类别索引张量描述。形状为 logits 的前 N-1 维（去掉最后一个类别维度），每个元素为类别索引（范围 [0, C-1]）。

参数限制：

- `loss` 和 `logits` 的 `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- `target` 的 `dT`: `Int32`。
- `logits` 的最后一个维度必须等于类别数 `C`。
- `target` 的形状必须与 `logits` 的前 N-1 维相同。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetCrossEntropyLossWorkspaceSize(
    infiniopCrossEntropyLossDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 已使用 `infiniopCreateCrossEntropyLossDescriptor()` 初始化的算子描述符；
- `size`: 额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyCrossEntropyLossDescriptor(
    infiniopCrossEntropyLossDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

无

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_NULL_POINTER`]: /common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
