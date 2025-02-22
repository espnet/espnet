import type { HeadConfig } from 'vuepress/core'

export const head: HeadConfig[] = [
  ['link', { rel: 'manifest', href: '/manifest.webmanifest' }],
  ['meta', { name: 'application-name', content: 'Example' }],
  ['meta', { name: 'apple-mobile-web-app-title', content: 'Example' }],
  ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }],
  ['meta', { name: 'msapplication-TileColor', content: '#3eaf7c' }],
  ['meta', { name: 'theme-color', content: '#3eaf7c' }],
]
