--[
-- Various utility functions.
--
-- (c) 2014 Brandon L. Reiss
--]

--- Convert a string of binary data to an integer.
local function string_to_int(str)
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

--- Convert a value into a string of bytes length.
local function int_to_string(value, bytes)

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

--- Convert a sequence of bytes to a binary string.
--- @param data_arr a table of pairs {{#bytes, value}, ...}
local function data_to_binary_string(data_arr)

    local binary = ''

    for _, pair in ipairs(data_arr) do
        bytes, value = unpack(pair)
        binary = binary..int_to_string(value, bytes)
    end

    return binary
end

--- A file interface backed by a string.
local function string_io(initial_data)
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

util = {
    string_to_int = string_to_int,
    int_to_string = int_to_string,
    data_to_binary_string = data_to_binary_string,
    string_io = string_io,
}
return util
