#!/bin/sh
#

git filter-branch --force --env-filter '
    if [ "$GIT_COMMITTER_NAME" = "tanghaoyu" ];
    then
        GIT_COMMITTER_NAME="qmpzzpmq";
        GIT_COMMITTER_EMAIL="405691733@qq.com";
        GIT_AUTHOR_NAME="qmpzzpmq";
        GIT_AUTHOR_EMAIL="405691733@qq.com";
    fi' -- --all
