import { defineConfig } from 'tsdown'

// 宿主型插件（无浏览器端 client）：输出单一 ESM 包 + 类型声明。
// 运行时零 npm 依赖（仅 Node 内置 child_process），因此所有外部包保持 external。
export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm'],
  outDir: 'lib',
  dts: true,
  clean: true,
  // 运行时仅依赖 Node 内置模块与 @deepseek-ai 类型（type-only，构建即擦除），
  // 不打包任何第三方运行时包。
  deps: {
    neverBundle: ['@deepseek-ai/cordis', '@deepseek-ai/dsh-tools', 'node:*'],
  },
  // 强制输出 .js / .d.ts（与 package.json 的 main/types 对齐），
  // 否则 tsdown 在 ESM 输出下默认写成 .mjs / .d.mts。
  outExtensions: () => ({ js: '.js', dts: '.d.ts' }),
})
