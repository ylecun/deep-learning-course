#!/bin/bash

test_module() {
    mod_name="${1%.*}"

    echo '***************************************'
    echo Testing module: ${mod_name}
    th -e "m = require(\"${mod_name}\"); m._test()"
    echo
    echo
}

# When there are arguments, use them as paths.
if [[ $# > 0 ]]; then
    for script in $@; do
    test_module ${script}
    done

else
    script_dir=`dirname $0`
    lua_dir="${script_dir}/../lua"

    pushd "${lua_dir}"

    # Find lua scripts.
    for script in `find . -type f -name '*.lua'`; do
        test_module ${script}
    done

    popd
fi

exit 0
