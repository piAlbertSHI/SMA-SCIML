# Release policy / 发布规范

SMA-SCIML remains private during release preparation. Presentation quality, scientific evidence, licensing and publication authorization are separate requirements. This policy does not authorize a visibility change.

SMA-SCIML 在发布准备期间保持私有。展示质量、科学证据、许可及发布授权是独立要求，本规范不授予可见性变更权限。

## Release requirements / 发布要求

- [ ] Confirm ownership and institutional/collaborator permission for code, data, figures and checkpoints. / 核实代码、数据、图像及权重的所有权和机构、合作者许可。
- [ ] Select explicit code, data and weight licenses; no license is implied by repository access. / 明确代码、数据及权重许可，仓库访问不代表再分发许可。
- [ ] Review complete Git history, attachments and release assets for credentials, personal information and unpublished material. / 审查完整 Git 历史、附件及发布资产中的凭据、个人信息和未发表资料。
- [ ] Verify installation, CPU inference and documented examples in a clean environment. / 在干净环境验证安装、CPU 推理与文档示例。
- [ ] Complete browser rendering and interaction acceptance for the offline explorer; mock DOM tests alone are insufficient. / 完成离线浏览器的真实渲染与交互验收，模拟 DOM 测试不能替代。
- [ ] Verify model/data cards, all method names, figure provenance and scientific limitations. / 核验模型与数据说明、方法命名、图表来源及科学限制。
- [ ] Document contributions and assistance accurately without inferring individual expertise from generated artifacts. / 准确记录贡献与辅助方式，不从生成的产物推断个人能力。
- [ ] Decide which historical development notes belong in the public distribution. / 确定哪些历史开发笔记纳入公开分发。
- [ ] Obtain explicit owner approval before changing visibility or enabling public hosting. / 获得所有者明确批准后方可公开或启用公共托管。

Current checks cover distribution hashes, known sensitive-text patterns, model contracts, checkpoint replay and presentation terminology. These checks do not replace ownership review, an exhaustive historical security audit or independent experimental validation.

现有检查覆盖分发哈希、已知敏感文本模式、模型接口、权重回放及展示术语，不替代权属审查、完整历史安全审计或独立实验验证。
