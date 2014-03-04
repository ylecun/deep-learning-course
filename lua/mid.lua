--[
-- Lua library for reading and writing MIDI (.mid) files.
--
-- An example usage is:
--
--   mid = require "mid"
--   data = mid.open(PATH_TO_MID_FILE)
--   -- From here, it's possible to access all members of the MID data from the
--   -- fields of 'data'. It's best to explore in a lua REPL.
--
--]
require "io"

--- The MIDI file header.
local HEADER, TRACK_HEADER = "MThd", "MTrk"

local MODE_SINGLE_TRACK, MODE_MULTI_SYNCH, MODE_MULTI_ASYNCH = 0, 1, 2

--- Convert a string of binary data to an integer.
local function string_to_int(str)
    local num = 0
    for idx = 1,#str do
        shift = 8 * (#str - idx)
        byte = string.byte(str, idx, idx)
        --print('num=' .. num .. ', idx=' .. idx .. ', shift=' .. shift ..
        --    ', byte=' .. byte .. ', val=' .. bit.lshift(byte, shift))
        num = num + bit.lshift(byte, shift)
    end
    return num
end

--- Mask for one byte.
local BYTE_MASK = 0xff

--- Convert a sequence of bytes to a binary string.
--- @param data_arr a table of pairs {{#bytes, value}, ...}
local function data_to_binary_string(data_arr)

    local binary = ''

    for _, pair in ipairs(data_arr) do
        bytes, value = unpack(pair)
        if bytes > 4 then
            error({msg="An int is at most 4 bytes"})
        end

        -- Convert value bytes to binary one by one.
        for idx = bytes,1,-1 do
            shift = 8 * (idx - 1)
            mask = bit.lshift(BYTE_MASK, shift)
            byte = bit.rshift(bit.band(value, mask), shift)
            binary = binary .. string.char(byte)
        end
    end

    return binary
end

--- Check length matches expected.
local function get_len_check_expected(midfile, num_bytes, expected)
    local len = string_to_int(midfile:read(num_bytes))
    if len ~= expected then
        error({msg=string.format(
                "Data field len %d != expected len %d",
                len, expected)})
    end
    return len
end

--- Process the track header.
local function process_track_header(midfile)

    -- Get track header magic number.
    local track_header = midfile:read(4)
    if track_header == nil then
        return nil
    elseif track_header ~= TRACK_HEADER then
        error({msg=string.format("MIDI track header 0x%x != expected 0x%x",
                track_header, TRACK_HEADER)})
    end

    return {
        size = string_to_int(midfile:read(4)),
    }
end

--- Read 7-bits-per-bytes variable length field.
local function read_var_len_value(midfile)

    -- Read variable-sized value in 7-bit bytes.
    local num_bytes, value = 0, 0
    repeat
        -- Read next byte and decrement remaining.
        local byte_raw = string_to_int(midfile:read(1))
        num_bytes = num_bytes + 1

        -- Shift 7-bit byte into place.
        local byte = bit.band(byte_raw, 0x7f)
        value = bit.lshift(value, 7)
        value = value + byte

        --print("byte_raw="..byte_raw..", shift="
        --        ..shift..", value="..value..", byte="..byte)

        -- Break when msb is 0.
        if byte_raw < 128 then
            break
        end
    until not true

    return num_bytes, value
end

--- Encode value with 7-bit bytes.
local function value_to_var_len_encoding(value)

    local binary = ""

    for shift = 21,7,-7 do
        local byte = bit.band(bit.rshift(value, shift), 0x7f)
        if byte > 0 or string.len(binary) > 0 then
            binary = binary .. string.char(bit.bor(0x80, byte))
        end
        --print("shift="..shift..", byte="..byte
        --        ..", string.len(binary)="..string.len(binary))
    end
    local lsb = bit.band(value, 0x7f)
    binary = binary .. string.char(lsb)
    --print("shift=0, byte="..lsb..", string.len(binary)="..string.len(binary))

    return binary
end

--- Read an event header.
local function process_event_header(midfile)
    local delta_bytes, delta_time = read_var_len_value(midfile)
    return delta_bytes, { delta_time = delta_time, }
end

--- Process a meta-command with command flag 0xff.
local function process_meta_command(midfile, event)

    -- Get meta-command type.
    event.meta_command = string_to_int(midfile:read(1))

    -- Complete steps for a text command. There are several.
    local handle_text_event = function(midfile, event)
        local _, len = read_var_len_value(midfile)
        event.len = len
        event.text = midfile:read(event.len)

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.meta_command},
        }) .. value_to_var_len_encoding(event.len)
           .. event.text
        return event
    end

    if event.meta_command == 0x00 then
        event.common_name = "meta - set track sequence number"
        event.len = get_len_check_expected(midfile, 1, 2)
        event.sequence_number = string_to_int(midfile:read(2))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.meta_command},
            {1, event.len},
            {2, event.sequence_number},
        })

    elseif event.meta_command == 0x01 then
        event.common_name = "meta - text"
        handle_text_event(midfile, event)

    elseif event.meta_command == 0x02 then
        event.common_name = "meta - text copyright"
        handle_text_event(midfile, event)

    elseif event.meta_command == 0x03 then
        event.common_name = "meta - seq/track name"
        handle_text_event(midfile, event)

    elseif event.meta_command == 0x04 then
        event.common_name = "meta - track instrument name"
        handle_text_event(midfile, event)

    elseif event.meta_command == 0x05 then
        event.common_name = "meta - lyric"
        handle_text_event(midfile, event)

    elseif event.meta_command == 0x06 then
        event.common_name = "meta - marker"
        handle_text_event(midfile, event)

    elseif event.meta_command == 0x07 then
        event.common_name = "meta - cue point"
        handle_text_event(midfile, event)

    elseif event.meta_command == 0x2f then
        event.common_name = "meta - track end"
        event.data = string_to_int(midfile:read(1))
        if event.data ~= 0x00 then
            error({msg="Track end meta command had nonzero data: "
                    .. string.format("0x%x", event.data)})
        end

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.meta_command},
            {1, event.data},
        })

    elseif event.meta_command == 0x51 then
        event.common_name = "meta - set tempo"
        event.len = get_len_check_expected(midfile, 1, 3)
        event.ms_per_qrtr_note = string_to_int(midfile:read(3))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.meta_command},
            {1, event.len},
            {3, event.ms_per_qrtr_note},
        })

    elseif event.meta_command == 0x58 then
        event.common_name = "meta - time signature"
        event.len = get_len_check_expected(midfile, 1, 4)
        event.numerator = string_to_int(midfile:read(1))
        event.denominator = string_to_int(midfile:read(1))
        event.metro_ticks_per_click = string_to_int(midfile:read(1))
        event.n32_notes_per_qrtr = string_to_int(midfile:read(1))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.meta_command},
            {1, event.len},
            {1, event.numerator},
            {1, event.denominator},
            {1, event.metro_ticks_per_click},
            {1, event.n32_notes_per_qrtr},
        })

    elseif event.meta_command == 0x59 then
        event.common_name = "meta - key signature"
        event.len = get_len_check_expected(midfile, 1, 2)
        local sharps_and_flats = string_to_int(midfile:read(1))
        event.sharps = bit.rshift(sharps_and_flats, 4)
        event.flats = bit.band(sharps_and_flats, 0x0f)
        local major_and_minor = string_to_int(midfile:read(1))
        event.major = bit.rshift(major_and_minor, 4)
        event.minor = bit.band(major_and_minor, 0x0f)

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.meta_command},
            {1, event.len},
            {1, sharps_and_flats},
            {1, major_and_minor},
        })

    elseif event.meta_command == 0x7f then
        event.common_name = "meta - sequencer specific"
        handle_text_event(midfile, event)

    else
        event.common_name = string.format(
                "Unrecognized meta-command: 0x%x", event.meta_command)
        handle_text_event(midfile, event)
    end

    return event
end

--- Process a system message. Does not need to read more data.
local function process_system_message(event)
    if event.command == 0xf8 then
        event.common_name = "timing clock"
    elseif event.command == 0xfa then
        event.common_name = "start current sequence"
    elseif event.command == 0xfb then
        event.common_name = "continue stopped sequence"
    elseif event.command == 0xfc then
        event.common_name = "stop sequence"
    else
        error({msg="Unrecognized system message: "
                .. string.format("0x%x", event.command)})
    end
    event.payload = string.char(event.command)

    return event
end

--- Process a channel event.
local function process_channel_event(midfile, event)

    -- Channel is lower nibble.
    event.channel = bit.band(event.command, 0x0f)

    if event.ctype == 0x8 then
        event.common_name = "note off"
        event.note_number = string_to_int(midfile:read(1))
        event.velocity = string_to_int(midfile:read(1))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.note_number},
            {1, event.velocity},
        })

    elseif event.ctype == 0x9 then
        event.common_name = "note on"
        event.note_number = string_to_int(midfile:read(1))
        event.velocity = string_to_int(midfile:read(1))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.note_number},
            {1, event.velocity},
        })

    elseif event.ctype == 0xa then
        event.common_name = "key after-touch"
        event.note_number = string_to_int(midfile:read(1))
        event.velocity = string_to_int(midfile:read(1))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.note_number},
            {1, event.velocity},
        })

    elseif event.ctype == 0xb then
        event.common_name = "control change"
        event.controller_num = string_to_int(midfile:read(1))
        event.new_value = string_to_int(midfile:read(1))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.controller_num},
            {1, event.new_value},
        })

    elseif event.ctype == 0xd then
        event.common_name = "program (patch) change"
        event.program_num = string_to_int(midfile:read(1))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.program_num},
        })

    elseif event.ctype == 0xd then
        event.common_name = "channel after-touch"
        event.channel_num = string_to_int(midfile:read(1))

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.channel_num},
        })

    elseif event.ctype == 0xe then
        event.common_name = "pitch wheel change"
        event.least_significant = string_to_int(midfile:read(1))
        event.most_significant = string_to_int(midfile:read(1))
        event.pitch = bit.lshift(bit.band(event.most_significant, 0x7f), 7)
                + bit.band(event.least_significant, 0x7f)

        event.payload = data_to_binary_string({
            {1, event.command},
            {1, event.least_significant},
            {1, event.most_significant},
        })

    else
        error({msg="Unrecognized command: "
                .. string.format("0x%x", event.command)})
    end
end

--- Open a .mid file.
local function open(path)
    local midfile = io.open(path)

    -- Perform file IO in protected block.
    --
    local status, err_or_data = pcall(function()

        -- Read .mid header.
        local header = midfile:read(4)
        if header ~= HEADER then
            error({msg=string.format("MIDI file header(%s) != expected %s",
                    header, HEADER)})
        end
        local header_size = get_len_check_expected(midfile, 4, 6)

        local data = {}
        data.track_mode = string_to_int(midfile:read(2))
        data.num_tracks = string_to_int(midfile:read(2))
        data.ticks_per_qrtr = string_to_int(midfile:read(2))
        data.tracks = {}

        -- Read tracks until EOF.
        repeat
            --print("reading track "..(#data.tracks + 1))

            -- Start a new track.
            local track = process_track_header(midfile)
            if track == nil then
                return data
            end
            --print("track:")
            --print(track)

            -- Read remainder of the track (the events).
            local bytes_remain = track.size
            local events = {}
            repeat
                --print("reading event "..(#events + 1))
                --print("bytes_remain="..bytes_remain)

                -- Create a new event.
                local delta_bytes, event = process_event_header(midfile)
                bytes_remain = bytes_remain - delta_bytes

                -- The command byte will become part of the payload, so don't
                -- deduct it yet.
                event.command = string_to_int(midfile:read(1))
                event.ctype = bit.rshift(event.command, 4)

                --print("delta_bytes="..delta_bytes..", command="..event.command)

                -- There are 3 types of events. See
                --   http://faydoc.tripod.com/formats/mid.htm
                -- for more information.
                if event.ctype == 0xf then
                    if event.command == 0xff then
                        process_meta_command(midfile, event)
                    else
                        process_system_message(event)
                    end
                else
                    process_channel_event(midfile, event)
                end

                -- Add event to list for this track.
                --print("event:")
                --print(event)
                table.insert(events, event)

                -- See if this track is read.
                bytes_remain = bytes_remain - string.len(event.payload)
                if bytes_remain == 0 then
                    --print("track "..(#data.tracks + 1).." completed")
                    break
                elseif bytes_remain < 0 then
                    error({msg=string.format("Track size is %d but read %d bytes",
                            track.size, track.size - bytes_remain)})
                end

            until not true

            track.events = events
            table.insert(data.tracks, track)

        until not true

    end)

    midfile.close()

    if not status then
        error(err_or_data.msg)
    else
        return err_or_data
    end
end

--- Write a .mid file.
local function writen(data)
end

--- Run tests.
local function _test()

    local function assert_equals(expected, actual)
        if actual ~= expected then
            print(debug.traceback())
            error("value expected="..tostring(expected)
                    ..", actual="..tostring(actual))
        end
    end

    local function check_test(test_func)
        status, err = pcall(test_func)
        if status == false then
            print("test error:")
            print(err)
        end
    end

    --- Create a mocked file from data.
    local function file_mock(data)
        return {
            data = data,
            pointer = 1,
            read = function(self, size)
                local tmp = self.pointer
                self.pointer = self.pointer + size
                local readval = self.data.sub(self.data, tmp, self.pointer - 1)
                --print("i="..tmp..", j="..self.pointer..", size="..size
                --        ..", len(readval)="..string.len(readval))
                return readval
            end
        }
    end

    -- Test variable size fields.
    check_test(function()

        -- Test simple value 0.
        local value = 0
        local binary = value_to_var_len_encoding(value)
        assert_equals(1, string.len(binary))
        local size, value_recovered = read_var_len_value(file_mock(binary))
        assert_equals(string.len(binary), size)
        assert_equals(value, value_recovered)

        -- Test values at borders of 7-bit bytes.
        for i = 1,3 do
            -- Do on low side of 7-bit number boundary.
            local value = bit.lshift(1, 7 * i) - 1
            local binary = value_to_var_len_encoding(value)
            assert_equals(i, string.len(binary))
            local size, value_recovered = read_var_len_value(file_mock(binary))
            assert_equals(string.len(binary), size)
            assert_equals(value, value_recovered)

            -- Do on high side of 7-bit number boundary.
            local value = bit.lshift(1, 7 * i)
            local binary = value_to_var_len_encoding(value)
            assert_equals(i + 1, string.len(binary))
            local size, value_recovered = read_var_len_value(file_mock(binary))
            assert_equals(string.len(binary), size)
            assert_equals(value, value_recovered)
        end

        -- Max value test.
        local value = bit.lshift(1, 7 * 4) - 1
        local binary = value_to_var_len_encoding(value)
        assert_equals(4, string.len(binary))
        local size, value_recovered = read_var_len_value(file_mock(binary))
        assert_equals(string.len(binary), size)
        assert_equals(value, value_recovered)
    end)
end

return {
    HEADER = HEADER,
    MODE_SINGLE_TRACK = MODE_SINGLE_TRACK,
    MODE_MULTI_SYNCH = MODE_MULTI_SYNCH,
    MODE_MULTI_ASYNCH = MODE_MULTI_ASYNCH,
    open = open,
    write = write,
    _test = _test,
}

