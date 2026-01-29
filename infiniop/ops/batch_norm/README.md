# `BatchNorm`

`BatchNorm`, 即**批归一化**算子，对输入张量进行批归一化操作，通过计算批次统计信息来归一化输入，常用于深度神经网络的训练和推理。

对于输入张量 $x$，批归一化的计算公式为：

$$ y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta $$

其中：
- $\mu$ 是运行均值（running mean）
- $\sigma^2$ 是运行方差（running variance）
- $\gamma$ 是可学习的缩放参数（weight）
- $\beta$ 是可学习的偏移参数（bias）
- $\epsilon$ 是数值稳定性参数（eps）

在训练过程中，运行均值和方差会根据当前批次的统计信息使用动量（momentum）进行更新。

## 接口

### 计算

```c
infiniStatus_t infiniopBatchNorm(
    infiniopBatchNormDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    void *running_mean,
    void *running_var,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 已使用 `infiniopCreateBatchNormDescriptor()` 初始化的算子描述符；
- `workspace`: 指向算子计算所需的额外工作空间；
- `workspace_size`: `workspace` 的大小，单位：字节；
- `output`: 输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `running_mean`: 运行均值张量，会被更新。张量限制见[创建算子描述](#创建算子描述)部分；
- `running_var`: 运行方差张量，会被更新。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`: 输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `weight`: 缩放参数张量（$\gamma$）。张量限制见[创建算子描述](#创建算子描述)部分；
- `bias`: 偏移参数张量（$\beta$）。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`: 计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateBatchNormDescriptor(
    infiniopHandle_t handle,
    infiniopBatchNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float momentum,
    float eps
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: `infiniopBatchNormDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | (N, C, L) | (...) }:
  输出张量描述。形状为 `(N, C, L)`，其中 `N` 为批次大小，`C` 为通道数，`L` 为空间维度大小。
- `running_mean_desc` - { dT | (C,) | (...) }:
  运行均值张量描述。形状为 `(C,)`，一维张量，长度为通道数。
- `running_var_desc` - { dT | (C,) | (...) }:
  运行方差张量描述。形状为 `(C,)`，一维张量，长度为通道数。
- `input_desc` - { dT | (N, C, L) | (...) }:
  输入张量描述。形状必须与输出相同。
- `weight_desc` - { dT | (C,) | (...) }:
  缩放参数张量描述。形状为 `(C,)`，一维张量，长度为通道数。
- `bias_desc` - { dT | (C,) | (...) }:
  偏移参数张量描述。形状为 `(C,)`，一维张量，长度为通道数。
- `momentum`: 动量参数，用于更新运行统计信息，范围通常在 [0, 1]。
- `eps`: 数值稳定性参数，添加到方差中以避免除零，通常为小的正数（如 1e-5）。

参数限制：

- `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- 输入和输出张量的形状必须相同，且维度必须为 3（1D BatchNorm）。
- `running_mean`、`running_var`、`weight`、`bias` 必须是一维张量，长度等于通道数 `C`。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetBatchNormWorkspaceSize(
    infiniopBatchNormDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 已使用 `infiniopCreateBatchNormDescriptor()` 初始化的算子描述符；
- `size`: 额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyBatchNormDescriptor(
    infiniopBatchNormDescriptor_t desc
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
