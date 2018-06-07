#ifndef CUTORCH_UTILS_INC
#define CUTORCH_UTILS_INC

#include "luaT.h"
#include "TH.h"

#ifdef __cplusplus
# define TORCH_EXTERNC extern "C"
#else
# define TORCH_EXTERNC extern
#endif

#ifdef __GNUC__
# define TORCH_UNUSED __attribute__((unused))
#else
# define TORCH_UNUSED
#endif

#ifdef _WIN32
# ifdef torch_EXPORTS
#  define TORCH_API TORCH_EXTERNC __declspec(dllexport)
# else
#  define TORCH_API TORCH_EXTERNC __declspec(dllimport)
# endif
#else
# define TORCH_API TORCH_EXTERNC
#endif

#if LUA_VERSION_NUM == 501
/*
** Adapted from Lua 5.2.0
*/
TORCH_UNUSED static void luaL_setfuncs (lua_State *L, const luaL_Reg *l, int nup) {
    luaL_checkstack(L, nup+1, "too many upvalues");
    for (; l->name != NULL; l++) {  /* fill the table with given functions */
        int i;
        lua_pushstring(L, l->name);
        for (i = 0; i < nup; i++)  /* copy upvalues to the top */
            lua_pushvalue(L, -(nup+1));
        lua_pushcclosure(L, l->func, nup);  /* closure with those upvalues */
        lua_settable(L, -(nup + 3));
    }
    lua_pop(L, nup);  /* remove upvalues */
}
#endif


TORCH_API THLongStorage* cutorch_checklongargs(lua_State *L, int index);
TORCH_API int cutorch_islongargs(lua_State *L, int index);

struct THCState;
TORCH_API struct THCState* cutorch_getstate(lua_State* L);

#endif