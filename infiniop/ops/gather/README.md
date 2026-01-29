# `Gather`

`Gather`, 即**收集**算子，根据索引张量从输入张量中收集元素，沿着指定维度进行索引操作。

对于输入张量 $x$ 和索引张量 $idx$，在维度 `dim` 上进行 gather 操作，输出张量 $y$ 的计算公式为：

$$ y_{i_1,...,i_{dim},...,i_n} = x_{i_1,...,idx_{i_1,...,i_n},...,i_n} $$

其中索引值 `idx` 指定了在维度 `dim` 上要收集的元素位置。

## 接口

### 计算

```c
infiniStatus_t infiniopGather(
    infiniopGatherDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *index,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 已使用 `infiniopCreateGatherDescriptor()` 初始化的算子描述符；
- `workspace`: 指向算子计算所需的额外工作空间；
- `workspace_size`: `workspace` 的大小，单位：字节；
- `output`: 输出张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `input`: 输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `index`: 索引张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`: 计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]，[`INFINI_STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateGatherDescriptor(
    infiniopHandle_t handle,
    infiniopGatherDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t index_desc,
    size_t dim
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`: `infiniopGatherDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `output_desc` - { dT | shape }:
  输出张量描述。形状与 `index_desc` 相同。
- `input_desc` - { dT | (d1,...,dn) | (...) }:
  输入张量描述。从该张量中收集元素。
- `index_desc` - { Int32/Int64 | shape }:
  索引张量描述。形状与输出相同，每个元素指定在维度 `dim` 上要收集的元素索引。
- `dim`: 指定进行 gather 操作的维度索引。

参数限制：

- `input` 和 `output` 的 `dT`: (`Float16`, `Float32`, `BFloat16`) 之一。
- `index` 的 `dT`: `Int32` 或 `Int64`。
- `dim` 必须在 [0, input.ndim-1] 范围内。
- `output` 的形状必须与 `index` 相同。
- `input` 和 `index` 在除 `dim` 维度外的所有维度上必须具有相同的形状。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetGatherWorkspaceSize(
    infiniopGatherDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`: 已使用 `infiniopCreateGatherDescriptor()` 初始化的算子描述符；
- `size`: 额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyGatherDescriptor(
    infiniopGatherDescriptor_t desc
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
