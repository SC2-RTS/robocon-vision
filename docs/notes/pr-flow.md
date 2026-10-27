# PR 流程笔记（D3 实操）

> 原文件误放在 `tests/try_PR`，已挪到这里。`tests/` 只放 pytest 的测试文件。

## 1. PR 是什么

PR（Pull Request）不是 git 命令，是 **GitHub 平台**的功能。含义是「我这条分支改好了，请求把它合并进 main」。

它的价值在**过程**：有独立的 diff 视图、能逐行评论、能记录谁审过、能挂自动测试。

## 2. 五个名词别混

| 名词 | 属于 | 一句话 |
|---|---|---|
| commit | Git | 一次代码快照 |
| branch | Git | 一条独立的开发线 |
| PR | GitHub | 请求把分支合并进 main 的提案 |
| tag | Git | 给某个 commit 起一个永不移动的名字（版本号） |
| Release | GitHub | 绑定某个 tag 的发布页面（变更说明 + 可下载文件） |

## 3. 完整顺序（顺序错了就会错位）

```text
1. 从最新 main 开分支，命名说明改什么（docs/readme、feat/xxx）
2. 小步 commit，带规范前缀（feat/fix/docs/chore/test）
3. push 分支
4. 网页开 PR：标题 + 改了什么 / 为什么 / 怎么验证
5. Files changed 逐行 review → Comment（自己不能 Approve 自己的 PR）
6. Merge pull request → Confirm merge → Delete branch
7. git switch main && git pull        ← 关键，否则本地落后
8. git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z
9. Releases → Draft a new release → 选 tag → Generate release notes → Publish
```

## 4. 踩过的坑（2026-10-27）

| 坑 | 现象 | 根因 | 正确做法 |
|---|---|---|---|
| tag 打在分支上 | `v0.2.0` 指向 `new_branch` 的提交 | 先打 tag 后合并 | **合并 → pull → 再打 tag** |
| 远程 tag 与本地不一致 | 本地 `07e60b1`，远程 `ccd73bf` | 推完 tag 又 amend/reset 了提交；tag 是死指针 | 删远程 + 删本地 + 重打 |
| 骨架目录没提交 | main 上只有 3 个文件 | **Git 不跟踪空目录** | 每个空目录放 `.gitkeep`，Python 包放 `__init__.py` |
| 远端合并后本地不同步 | 本地 main 停在旧位置 | 忘了 `git pull` | 合并后第一件事就是 pull |
