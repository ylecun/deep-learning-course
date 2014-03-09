--[
-- The .mid file event parser. The read() and write() aspects are placed next
-- to one another for readability since they must be parallel routines.
--
-- (c) 2014 Brandon L. Reiss
--]
require "bit"
require "util"
require "test_util"

local string_to_int = util.string_to_int
local int_to_string = util.int_to_string
local data_to_binary_string = util.data_to_binary_string

event = {}

--- The command type names.
event.CODES = {
    -- Meta commands are subtypes of the meta code.
    meta_prefix = 0xff,
    meta = {
        set_trk_seq_num = 0x00,
        text = 0x01,
        text_copyright = 0x02,
        seq_track_name = 0x03,
        track_instrument_name = 0x04,
        lyric = 0x05,
        marker = 0x06,
        cue_point = 0x07,
        track_end = 0x2f,
        set_tempo = 0x51,
        time_sig = 0x58,
        key_sig = 0x59,
        seq_specific = 0x7f,
    },

    -- System codes are at the top level and have no channel.
    system = {
        timing_clock = 0xf8,
        start_cur_seq = 0xfa,
        cont_stop_seq = 0xfb,
        stop_seq = 0xfc,
    },

    -- Commands with channel nibble as lower 4 bits.
    midi = {
        note_off = 0x8,
        note_on = 0x9,
        key_after_touch = 0xa,
        control_change = 0xb,
        prog_change = 0xd,
        chan_after_touch = 0xd,
        pitch_wheel_change = 0xe,
    },
}
local CODES = event.CODES

--- Decode 7-bits-per-bytes variable length field.
function event.decode_var_len_value(file)

    -- Read variable-sized value in 7-bit bytes.
    local num_bytes, value = 0, 0
    repeat
        -- Read next byte and decrement remaining.
        local byte_raw = string_to_int(file:read(1))
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
local decode_var_len_value = event.decode_var_len_value

--- Encode value with 7-bit bytes.
function event.encode_var_len_value(value)

    local binary = ""

    for shift = 21,7,-7 do
        local byte = bit.band(bit.rshift(value, shift), 0x7f)
        if byte > 0 or binary:len() > 0 then
            binary = binary .. string.char(bit.bor(0x80, byte))
        end
        --print("shift="..shift..", byte="..byte
        --        ..", binary:len()="..binary:len())
    end
    local lsb = bit.band(value, 0x7f)
    binary = binary .. string.char(lsb)
    --print("shift=0, byte="..lsb..", binary:len()="..binary:len())

    return binary
end
local encode_var_len_value = event.encode_var_len_value

--- Check length matches expected.
function event.get_len_check_expected(file, num_bytes, expected)
    local len = string_to_int(file:read(num_bytes))
    if len ~= expected then
        error({msg=string.format(
                "Data field len %d != expected len %d",
                len, expected)})
    end
    return len
end
local get_len_check_expected = event.get_len_check_expected

--- Read an event containing only a text field.
local function text_event_read(file, event)
    local len_bytes, len = decode_var_len_value(file)
    event.len = len
    event.text = file:read(event.len)
    return len_bytes + len
end

--- Convert a text-only event to a binary string.
local function text_event_write(event, file)
    return file:write(encode_var_len_value(event.len)..event.text)
end

--- The event parser.
event.parser = {

    [CODES.meta_prefix] = {

        [CODES.meta.set_trk_seq_num] = {
            read = function(self, file, event)
                event.common_name = "meta - set track sequence number"
                event.len = get_len_check_expected(file, 1, 2)
                event.sequence_number = string_to_int(file:read(2))
                return 3
            end,
            write = function(self, event, file)
                return file:write(data_to_binary_string({
                    {1, event.len},
                    {event.len, event.sequence_number},
                }))
            end
        },

        [CODES.meta.text] = {
            read = function(self, file, event)
                event.common_name = "meta - text"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        [CODES.meta.text_copyright] = {
            read = function(self, file, event)
                event.common_name = "meta - text copyright"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        [CODES.meta.seq_track_name] = {

            read = function(self, file, event)
                event.common_name = "meta - seq/track name"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        [CODES.meta.track_instrument_name] = {
            read = function(self, file, event)
                event.common_name = "meta - track instrument name"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        [CODES.meta.lyric] = {
            read = function(self, file, event)
                event.common_name = "meta - lyric"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        [CODES.meta.marker] = {
            read = function(self, file, event)
                event.common_name = "meta - marker"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        [CODES.meta.cue_point] = {
            read = function(self, file, event)
                event.common_name = "meta - cue point"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        [CODES.meta.track_end] = {
            read = function(self, file, event)
                event.common_name = "meta - track end"
                event.data = string_to_int(file:read(1))
                if event.data ~= 0x00 then
                    error({msg=string.format(
                            "Track end meta command had nonzero data: 0x%x",
                            event.data)})
                end
                return 1
            end,
            write = function(self, event, file)
                return file:write(int_to_string(event.data, 1))
            end
        },

        [CODES.meta.set_tempo] = {
            read = function(self, file, event)
                event.common_name = "meta - set tempo"
                event.len = get_len_check_expected(file, 1, 3)
                event.us_per_qrtr_note = string_to_int(file:read(3))
                return 4
            end,
            write = function(self, event, file)
                return file:write(data_to_binary_string({
                    {1, event.len},
                    {3, event.us_per_qrtr_note},
                }))
            end
        },

        [CODES.meta.time_sig] = {
            read = function(self, file, event)
                event.common_name = "meta - time signature"
                event.len = get_len_check_expected(file, 1, 4)
                event.numerator = string_to_int(file:read(1))
                event.denominator = string_to_int(file:read(1))
                event.metro_ticks_per_click = string_to_int(file:read(1))
                event.n32_notes_per_qrtr = string_to_int(file:read(1))
                return 5
            end,
            write = function(self, event, file)
                return file:write(data_to_binary_string({
                    {1, event.len},
                    {1, event.numerator},
                    {1, event.denominator},
                    {1, event.metro_ticks_per_click},
                    {1, event.n32_notes_per_qrtr},
                }))
            end
        },

        [CODES.meta.key_sig] = {
            read = function(self, file, event)
                event.common_name = "meta - key signature"
                event.len = get_len_check_expected(file, 1, 2)
                local sharps_and_flats = string_to_int(file:read(1))
                event.sharps = bit.rshift(sharps_and_flats, 4)
                event.flats = bit.band(sharps_and_flats, 0x0f)
                local major_and_minor = string_to_int(file:read(1))
                event.major = bit.rshift(major_and_minor, 4)
                event.minor = bit.band(major_and_minor, 0x0f)
                return 3
            end,
            write = function(self, event, file)
                local sharps_and_flats =
                        bit.lshift(event.sharps, 4) + event.flats
                local major_and_minor =
                        bit.lshift(event.major, 4) + event.minor
                return file:write(data_to_binary_string({
                    {1, event.len},
                    {1, sharps_and_flats},
                    {1, major_and_minor},
                }))
            end
        },

        [CODES.meta.seq_specific] = {
            read = function(self, file, event)
                event.common_name = "meta - sequencer specific"
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        undef = {
            read = function(self, file, event)
                event.common_name = string.format(
                        "Unrecognized meta-command: 0x%x", event.meta)
                return text_event_read(file, event)
            end,
            write = function(self, event, file)
                return text_event_write(event, file)
            end
        },

        read = function(self, file, event)
            event.meta = string_to_int(file:read(1))

            --print(string.format("meta=0x%x", event.meta))

            local meta_parser = self[event.meta] or self.undef
            local bytes_read = meta_parser:read(file, event)

            return 1 + bytes_read
        end,

        write = function(self, event, file)
            local meta_parser = self[event.meta] or self.undef
            return file:write(int_to_string(event.meta, 1))
                    and meta_parser:write(event, file)
        end,
    },

    [CODES.system.timing_clock] = {
        read = function(self, file, event)
            event.common_name = "system - timing clock"
            return 0
        end,
        write = function(self, event, file) end
    },

    [CODES.system.start_cur_seq] = {
        read = function(self, file, event)
            event.common_name = "system - start current sequence"
            return 0
        end,
        write = function(self, event, file) end
    },

    [CODES.system.cont_stop_seq] = {
        read = function(self, file, event)
            event.common_name = "system - continue stopped sequence"
            return 0
        end,
        write = function(self, event, file) end
    },

    [CODES.system.stop_seq] = {
        read = function(self, file, event)
            event.common_name = "system - stop sequence"
            return 0
        end,
        write = function(self, event, file) end
    },

    [CODES.midi.note_off] = {
        read = function(self, file, event)
            event.common_name = "midi - note off"
            event.note_number = string_to_int(file:read(1))
            event.velocity = string_to_int(file:read(1))
            return 2
        end,
        write = function(self, event, file)
            return file:write(data_to_binary_string({
                {1, event.note_number},
                {1, event.velocity},
            }))
        end
    },

    [CODES.midi.note_on] = {
        read = function(self, file, event)
            event.common_name = "midi - note on"
            event.note_number = string_to_int(file:read(1))
            event.velocity = string_to_int(file:read(1))
            return 2
        end,
        write = function(self, event, file)
            return file:write(data_to_binary_string({
                {1, event.note_number},
                {1, event.velocity},
            }))
        end
    },

    [CODES.midi.key_after_touch] = {
        read = function(self, file, event)
            event.common_name = "midi - key after-touch"
            event.note_number = string_to_int(file:read(1))
            event.velocity = string_to_int(file:read(1))
            return 2
        end,
        write = function(self, event, file)
            return file:write(data_to_binary_string({
                {1, event.note_number},
                {1, event.velocity},
            }))
        end
    },

    [CODES.midi.control_change] = {
        read = function(self, file, event)
            event.common_name = "midi - control change"
            event.controller_num = string_to_int(file:read(1))
            event.new_value = string_to_int(file:read(1))
            return 2
        end,
        write = function(self, event, file)
            return file:write(data_to_binary_string({
                {1, event.controller_num},
                {1, event.new_value},
            }))
        end
    },

    [CODES.midi.prog_change] = {
        read = function(self, file, event)
            event.common_name = "midi - program (patch) change"
            event.program_num = string_to_int(file:read(1))
            return 1
        end,
        write = function(self, event, file)
            return file:write(int_to_string(event.program_num, 1))
        end
    },

    [CODES.midi.chan_after_touch] = {
        read = function(self, file, event)
            event.common_name = "midi - channel after-touch"
            event.channel_num = string_to_int(file:read(1))
            return 1
        end,
        write = function(self, event, file)
            return file:write(int_to_string(event.channel_num, 1))
        end
    },

    [CODES.midi.pitch_wheel_change] = {
        read = function(self, file, event)
            event.common_name = "midi - pitch wheel change"
            event.least_significant = string_to_int(file:read(1))
            event.most_significant = string_to_int(file:read(1))
            event.pitch = bit.lshift(bit.band(event.most_significant, 0x7f), 7)
                    + bit.band(event.least_significant, 0x7f)
            return 2
        end,
        write = function(self, event, file)
            return file:write(data_to_binary_string({
                {1, event.least_significant},
                {1, event.most_significant},
            }))
        end
    },

    undef = {
        read = function(self, file, event)
        event.common_name = string.format(
                "Unrecognized command: 0x%x", event.command)
            return text_event_read(file, event)
        end,
        write = function(self, event, file)
            return text_event_write(event, file)
        end
    },

    read = function(self, file, event)

        -- Read variable length delta time field.
        local delta_bytes, delta_time = decode_var_len_value(file)
        event.delta_time = delta_time

        event.command = string_to_int(file:read(1))
        local midi = bit.rshift(event.command, 4)

        --print(string.format("command=0x%x", event.command))

        local command_parser
        if self[event.command] ~= nil then
            command_parser = self[event.command]
        elseif self[midi] ~= nil then
            event.midi = midi
            event.channel = bit.band(event.command, 0x0f)
            command_parser = self[midi]
        else
            command_parser = self.undef
        end
        local bytes_read = command_parser:read(file, event)

        return delta_bytes + 1 + bytes_read
    end,

    write = function(self, event, file)
        local command_parser = self[event.command]
                or self[event.midi]
                or self.undef
        return file:write(encode_var_len_value(event.delta_time)
                .. int_to_string(event.command, 1))
                and command_parser:write(event, file)
    end,
}
local parser = event.parser

--- Run tests.
function event._test()

    local string_io = util.string_io
    local assert_equals = test_util.assert_equals
    local check_test = test_util.check_test

    -- Test variable size fields.
    check_test(function()

        -- Test simple value 0.
        local value = 0
        local binary = encode_var_len_value(value)
        assert_equals(1, binary:len())
        local size, value_recovered = decode_var_len_value(string_io(binary))
        assert_equals(binary:len(), size)
        assert_equals(value, value_recovered)

        -- Test values at borders of 7-bit bytes.
        for i = 1,3 do
            -- Do on low side of 7-bit number boundary.
            local value = bit.lshift(1, 7 * i) - 1
            local binary = encode_var_len_value(value)
            assert_equals(i, binary:len())
            local size, value_recovered = decode_var_len_value(string_io(binary))
            assert_equals(binary:len(), size)
            assert_equals(value, value_recovered)

            -- Do on high side of 7-bit number boundary.
            local value = bit.lshift(1, 7 * i)
            local binary = encode_var_len_value(value)
            assert_equals(i + 1, binary:len())
            local size, value_recovered = decode_var_len_value(string_io(binary))
            assert_equals(binary:len(), size)
            assert_equals(value, value_recovered)
        end

        -- Max value test.
        local value = bit.lshift(1, 7 * 4) - 1
        local binary = encode_var_len_value(value)
        assert_equals(4, binary:len())
        local size, value_recovered = decode_var_len_value(string_io(binary))
        assert_equals(binary:len(), size)
        assert_equals(value, value_recovered)
    end)

    -- Test event parsers.
    check_test(function()

        -- Create a simple event.
        local event = {}
        event.delta_time = 1021
        event.command = 0xff
        event.meta = 0x2f
        event.data = 0
        print("test event:")
        print(event)

        -- Serialize the simple event.
        local file = string_io()
        local result = parser:write(event, file)
        local from_binary = {}
        parser:read(string_io(file.data), from_binary)
        print("from_binary:")
        print(from_binary)

        -- Check equality.
        for key, value in pairs(event) do
            assert_equals(value, from_binary[key])
        end

        -- Serialize and read again since we only simulated essential
        -- fields.
        local file = string_io()
        local result = parser:write(from_binary, file)
        local from_binary_again = {}
        parser:read(string_io(file.data), from_binary_again)

        -- Check equality.
        for key, value in pairs(from_binary) do
            assert_equals(value, from_binary_again[key])
        end

    end)
end

return event

