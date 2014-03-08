#!/bin/bash

script_dir=`dirname $0`
lua_dir="${script_dir}/../lua"

pushd "${lua_dir}"

# Find lua scripts.
for script in `find . -type f -name '*.lua'`; do
    mod_name="${script%.*}"

    echo '***************************************'
    echo Testing module: ${mod_name}
    th -e "m = require(\"${mod_name}\"); m._test()"
    echo
    echo
done

popd

exit 0
