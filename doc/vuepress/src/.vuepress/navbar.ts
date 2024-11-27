import { navbar } from "vuepress-theme-hope";
import { load } from 'js-yaml'
import { readFileSync } from 'fs'

const navbarContents = load(readFileSync('navbars.yml', 'utf-8'))

export default navbar(navbarContents);
