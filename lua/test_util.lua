--[
-- Testing function utils.
--
-- (c) 2014 Brandon L. Reiss
--]

test_util = {}

--- Simple equality assertion. Types must be comparable with ~=.
function test_util.assert_equals(_msg, _expected, _actual)
    msg = _actual and _msg or nil
    expected = _actual and _expected or _msg
    actual = _actual or _expected 
    if actual ~= expected then
        local msg_prefix = ""
        if nill ~= msg then
            msg_prefix = "error: "..msg..", "
        end
        error(msg_prefix
                .."value expected="..tostring(expected)
                ..", actual="..tostring(actual))
    end
end

function test_util.assert_true(_msg, _actual)
    msg = _actual and _msg or nil
    actual = _actual or _msg 
    if not actual then
        local msg_prefix = ""
        if nill ~= msg then
            msg_prefix = "error: "..msg..", "
        end
        error(msg_prefix.."expecting true but got false")
    end
end

function test_util.assert_false(_msg, _actual)
    msg = _actual and _msg or nil
    actual = _actual or _msg 
    test_util.assert_true(msg, not actual)
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
