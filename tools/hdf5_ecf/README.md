# FRED HDF5 ECF 解码插件

原始 FRED 的 `events.hdf5` 使用 HDF5 filter `36559`。本目录为相应解码插件
保留固定的本地落点：

```text
tools/hdf5_ecf/plugin/
├── libH5Zecf.so
└── libhdf5_ecf_codec.so
```

两个动态库必须放在同一目录，`libH5Zecf.so` 通过 `$ORIGIN` 查找
`libhdf5_ecf_codec.so`。预处理脚本会自动检查上述位置，也可以通过
`--hdf5-plugin-path` 指定其他目录。

插件二进制仅作为本机运行依赖保存，已被 `.gitignore` 排除，不提交到公开
仓库。目前没有同时找到对应源码和明确的再发布许可；跨系统使用时应重新验证
兼容性，条件允许时优先从有来源记录的源码重新构建。
