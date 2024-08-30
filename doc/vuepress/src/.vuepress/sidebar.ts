import { sidebar } from "vuepress-theme-hope";
import { load } from 'js-yaml'
import { readFileSync } from 'fs'

const sidebarContent = load(readFileSync('sidebars.yml', 'utf-8'))

export default sidebar({
  "/": sidebarContent,
});
