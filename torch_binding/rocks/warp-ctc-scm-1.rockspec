package = "warp-ctc"
version = "scm-1"

source = {
   url = "git://github.com/baidu-research/warp-ctc.git",
}

description = {
   summary = "Baidu CTC Implementation",
   detailed = [[
   ]],
   homepage = "https://github.com/baidu-research/warp-ctc",
   license = "Apache"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) && make install
]],
	platforms = {},
   install_command = "cd build"
}
