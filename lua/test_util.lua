--[
-- Testing function utils.
--
-- (c) 2014 Brandon L. Reiss
--]

test_util = {}

--- Simple equality assertion. Types must be comparable with ~=.
function test_util.assert_equals(expected, actual)
    if actual ~= expected then
        error("value expected="..tostring(expected)
                ..", actual="..tostring(actual))
    end
end

--- Test wrapper that catches errors and prints stack traces.
function test_util.check_test(test_func)
    status, err = pcall(test_func)
    if status == false then
        print("test error:")
        print(err)
        print(debug.traceback())
    end
end

return test_util
