--[
-- Various utility functions.
--
-- (c) 2014 Brandon L. Reiss
--]
require "math"

require "test_util"

util = {}

--- Convert a string of binary data to an integer.
function util.string_to_int(str)
    local num = 0
    for idx = 1,#str do
        shift = 8 * (#str - idx)
        byte = str:byte(idx, idx)
        --print('num=' .. num .. ', idx=' .. idx .. ', shift=' .. shift ..
        --    ', byte=' .. byte .. ', val=' .. bit.lshift(byte, shift))
        num = num + bit.lshift(byte, shift)
    end
    return num
end
local string_to_int = util.string_to_int

--- Convert a value into a string of bytes length.
function util.int_to_string(value, bytes)

    if bytes > 4 then
        error({msg="An int is at most 4 bytes"})
    end

    -- Convert value bytes to binary one by one.
    local binary = ""
    for idx = bytes,1,-1 do
        shift = 8 * (idx - 1)
        mask = bit.lshift(0xff, shift)
        byte = bit.rshift(bit.band(value, mask), shift)
        binary = binary .. string.char(byte)
    end

    return binary
end
local int_to_string = util.int_to_string

--- Convert a sequence of bytes to a binary string.
--- @param data_arr a table of pairs {{#bytes, value}, ...}
function util.data_to_binary_string(data_arr)

    local binary = ''

    for _, pair in ipairs(data_arr) do
        bytes, value = unpack(pair)
        binary = binary..int_to_string(value, bytes)
    end

    return binary
end

--- A file interface backed by a string.
function util.string_io(initial_data)
    return {
        data = initial_data or "",
        pointer = 1,
        read = function(self, size)
            local tmp = self.pointer
            self.pointer = self.pointer + size
            local readval = self.data.sub(self.data, tmp, self.pointer - 1)
            --print("i="..tmp..", j="..self.pointer..", size="..size
            --        ..", readval:len()="..readval:len())
            return readval
        end,
        write = function(self, value)
            self.data = self.data .. value
            return true
        end,
    }
end

--- Binary search a sorted table. Returns positive index when the value is in
-- the sorted table and a negative index whose negated value is the insertion
-- index to maintain the sorted state of the table.
function util.binary_search(sorted_values, cmd)
    -- Emtpy table?
    if #sorted_values == 0 then
        return -1
    end

    local i, j = 1, #sorted_values
    repeat
        -- Check midpoint.
        local mid = math.floor((i + j) / 2)
        local ins = mid
        if sorted_values[mid] == cmd then
            return mid
        elseif sorted_values[mid] > cmd then
            j = mid - 1
        else
            i = mid + 1
            ins = i
        end
        -- See if not found.
        if i > j then
            return -ins
        end
    until not true
end
local binary_search = util.binary_search

function util._test()

    local assert_equals = test_util.assert_equals
    local check_test = test_util.check_test

    -- Test binary_search.
    check_test(function()
        -- Emtpy table test.
        local values = {}
        binary_search(values, 100)

        -- Some ints.
        local values = {0, 7, 13, 22, 101, 230}
        for i = -5,235 do
            -- Simple linear search.
            local expected_idx
            for idx, each in ipairs(values) do
                expected_idx = -idx
                if i == each then
                    expected_idx = idx
                    break
                elseif i < each then
                    break
                end
                expected_idx = -idx - 1
            end
            local actual_idx = binary_search(values, i)
            assert_equals(expected_idx, actual_idx)
        end
    end)
end

return util
