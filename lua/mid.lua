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
require "bit"
require "io"
require "lfs"
require "math"

--- The MIDI header magic numbers.
local HEADER, TRACK_HEADER = "MThd", "MTrk"

--- MIDI file modes.
local MODE_SINGLE_TRACK, MODE_MULTI_SYNCH, MODE_MULTI_ASYNCH = 0, 1, 2

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

--- Check length matches expected.
local function get_len_check_expected(file, num_bytes, expected)
    local len = string_to_int(file:read(num_bytes))
    if len ~= expected then
        error({msg=string.format(
                "Data field len %d != expected len %d",
                len, expected)})
    end
    return len
end

--- Process the track header.
local function process_track_header(file)

    -- Get track header magic number.
    local track_header = file:read(4)
    if track_header == nil then
        return nil
    elseif track_header ~= TRACK_HEADER then
        error({msg=string.format("MIDI track header 0x%x != expected 0x%x",
                track_header, TRACK_HEADER)})
    end

    return {
        size = string_to_int(file:read(4)),
    }
end

--- Read 7-bits-per-bytes variable length field.
local function read_var_len_value(file)

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

--- Encode value with 7-bit bytes.
local function value_to_var_len_encoding(value)

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

--- Read an event containing only a text field.
local function text_event_read(file, event)
	local len_bytes, len = read_var_len_value(file)
	event.len = len
	event.text = file:read(event.len)
	return len_bytes + len
end

--- Convert a text-only event to a binary string.
local function text_event_tostring(event)
    return value_to_var_len_encoding(event.len) .. event.text
end

--- The event parser.
local event_parser = {

    [0xff] = {

        [0x00] = {
            read = function(self, file, event)
                event.common_name = "meta - set track sequence number"
                event.len = get_len_check_expected(file, 1, 2)
                event.sequence_number = string_to_int(file:read(2))
				return 3
            end,
            tostring = function(self, event)
                return data_to_binary_string({
                    {1, event.len},
                    {event.len, event.sequence_number},
                })
            end
        },

        [0x01] = {
            read = function(self, file, event)
                event.common_name = "meta - text"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        [0x02] = {
            read = function(self, file, event)
                event.common_name = "meta - text copyright"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        [0x03] = {

            read = function(self, file, event)
                event.common_name = "meta - seq/track name"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        [0x04] = {
            read = function(self, file, event)
                event.common_name = "meta - track instrument name"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        [0x05] = {
            read = function(self, file, event)
                event.common_name = "meta - lyric"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        [0x06] = {
            read = function(self, file, event)
                event.common_name = "meta - marker"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        [0x07] = {
            read = function(self, file, event)
                event.common_name = "meta - cue point"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        [0x2f] = {
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
            tostring = function(self, event)
                return int_to_string(event.data, 1)
            end
        },

        [0x51] = {
            read = function(self, file, event)
                event.common_name = "meta - set tempo"
                event.len = get_len_check_expected(file, 1, 3)
                event.ms_per_qrtr_note = string_to_int(file:read(3))
				return 4
            end,
            tostring = function(self, event)
                return data_to_binary_string({
                    {1, event.len},
                    {3, event.ms_per_qrtr_note},
                })
            end
        },

        [0x58] = {
            read = function(self, file, event)
                event.common_name = "meta - time signature"
                event.len = get_len_check_expected(file, 1, 4)
                event.numerator = string_to_int(file:read(1))
                event.denominator = string_to_int(file:read(1))
                event.metro_ticks_per_click = string_to_int(file:read(1))
                event.n32_notes_per_qrtr = string_to_int(file:read(1))
				return 5
            end,
            tostring = function(self, event)
                return data_to_binary_string({
                    {1, event.len},
                    {1, event.numerator},
                    {1, event.denominator},
                    {1, event.metro_ticks_per_click},
                    {1, event.n32_notes_per_qrtr},
                })
            end
        },

        [0x59] = {
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
            tostring = function(self, event)
                local sharps_and_flats =
                        bit.lshift(event.sharps, 4) + event.flats
                local major_and_minor =
                        bit.lshift(event.major, 4) + event.minor
                return data_to_binary_string({
                    {1, event.len},
                    {1, sharps_and_flats},
                    {1, major_and_minor},
                })
            end
        },

        [0x7f] = {
            read = function(self, file, event)
                event.common_name = "meta - sequencer specific"
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        undef = {
            read = function(self, file, event)
                event.common_name = string.format(
                        "Unrecognized meta-command: 0x%x", event.meta_command)
                return text_event_read(file, event)
            end,
            tostring = function(self, event)
                return text_event_tostring(event)
            end
        },

        read = function(self, file, event)
            event.meta_command = string_to_int(file:read(1))

			--print(string.format("meta_command=0x%x", event.meta_command))

            local meta_parser = self[event.meta_command] or self.undef
            local bytes_read = meta_parser:read(file, event)

            return 1 + bytes_read
        end,

        tostring = function(self, event)
            local meta_parser = self[event.meta_command] or self..undef
            return  int_to_string(event.meta_command, 1)
                    .. meta_parser:tostring(event)
        end,
    },

	[0xf8] = {
		read = function(self, file, event)
			event.common_name = "timing clock"
			return 0
		end,
        tostring = function(self, event) end
	},

	[0xfa] = {
		read = function(self, file, event)
			event.common_name = "start current sequence"
			return 0
		end,
        tostring = function(self, event) end
	},

	[0xfb] = {
		read = function(self, file, event)
			event.common_name = "continue stopped sequence"
			return 0
		end,
        tostring = function(self, event) end
	},

	[0xfc] = {
		read = function(self, file, event)
			event.common_name = "stop sequence"
			return 0
		end,
        tostring = function(self, event) end
	},

	[0x8] = {
		read = function(self, file, event)
			event.common_name = "note off"
			event.note_number = string_to_int(file:read(1))
			event.velocity = string_to_int(file:read(1))
			return 2
		end,
		tostring = function(self, event)
			return data_to_binary_string({
				{1, event.note_number},
				{1, event.velocity},
			})
		end
	},

	[0x9] = {
		read = function(self, file, event)
			event.common_name = "note on"
			event.note_number = string_to_int(file:read(1))
			event.velocity = string_to_int(file:read(1))
			return 2
		end,
		tostring = function(self, event)
			return data_to_binary_string({
				{1, event.note_number},
				{1, event.velocity},
			})
		end
	},

	[0xa] = {
		read = function(self, file, event)
			event.common_name = "key after-touch"
			event.note_number = string_to_int(file:read(1))
			event.velocity = string_to_int(file:read(1))
			return 2
		end,
		tostring = function(self, event)
			return data_to_binary_string({
				{1, event.note_number},
				{1, event.velocity},
			})
		end
	},

	[0xb] = {
		read = function(self, file, event)
			event.common_name = "control change"
			event.controller_num = string_to_int(file:read(1))
			event.new_value = string_to_int(file:read(1))
			return 2
		end,
		tostring = function(self, event)
			return data_to_binary_string({
				{1, event.controller_num},
				{1, event.new_value},
			})
		end
	},

	[0xd] = {
		read = function(self, file, event)
			event.common_name = "program (patch) change"
			event.program_num = string_to_int(file:read(1))
			return 1
		end,
		tostring = function(self, event)
			return int_to_string(event.program_num, 1)
		end
	},

	[0xd] = {
		read = function(self, file, event)
			event.common_name = "channel after-touch"
			event.channel_num = string_to_int(file:read(1))
			return 1
		end,
		tostring = function(self, event)
			return int_to_string(event.channel_num, 1)
		end
	},

	[0xe] = {
		read = function(self, file, event)
			event.common_name = "pitch wheel change"
			event.least_significant = string_to_int(file:read(1))
			event.most_significant = string_to_int(file:read(1))
			event.pitch = bit.lshift(bit.band(event.most_significant, 0x7f), 7)
					+ bit.band(event.least_significant, 0x7f)
			return 2
		end,
		tostring = function(self, event)
			event.payload = data_to_binary_string({
				{1, event.least_significant},
				{1, event.most_significant},
			})
		end
	},

	undef = {
		read = function(self, file, event)
		event.common_name = string.format(
				"Unrecognized command: 0x%x", event.command)
			return text_event_read(file, event)
		end,
		tostring = function(self, event)
			return text_event_tostring(event)
		end
	},

    read = function(self, file, event)

		-- Read variable length delta time field.
		local delta_bytes, delta_time = read_var_len_value(file)
		event.delta_time = delta_time

        event.command = string_to_int(file:read(1))
        local ctype = bit.rshift(event.command, 4)

		--print(string.format("command=0x%x", event.command))

        local command_parser
		if self[event.command] ~= nil then
			command_parser = self[event.command]
		elseif self[ctype] ~= nil then
			event.ctype = ctype
			event.channel = bit.band(event.command, 0x0f)
			command_parser = self[ctype]
		else
			command_parser = self.undef
		end
        local bytes_read = command_parser:read(file, event)

        return delta_bytes + 1 + bytes_read
    end,

    tostring = function(self, event)
        local command_parser = self[event.command]
                or self[event.ctype]
                or self.undef
        return value_to_var_len_encoding(event.delta_time)
				.. int_to_string(event.command, 1)
                .. command_parser:tostring(event)
    end,
}

--- Open a .mid file.
local function read(path)
    local file = io.open(path)

    -- Perform file IO in protected block.
    --
    local status, err_or_data = pcall(function()

        -- Read .mid header.
        local header = file:read(4)
        if header ~= HEADER then
            error({msg=string.format("MIDI file header(%s) != expected %s",
                    header, HEADER)})
        end
        local header_size = get_len_check_expected(file, 4, 6)

        local data = {}
        data.track_mode = string_to_int(file:read(2))
        data.num_tracks = string_to_int(file:read(2))
        data.ticks_per_qrtr = string_to_int(file:read(2))
        data.tracks = {}

        -- Read tracks until EOF.
        repeat
            --print("reading track "..(#data.tracks + 1))

            -- Start a new track.
            local track = process_track_header(file)
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

				local event = {}
				local bytes_read = event_parser:read(file, event)

                -- Add event to list for this track.
                --print("event:")
                --print(event)
                table.insert(events, event)

                -- See if this track is read.
                bytes_remain = bytes_remain - bytes_read
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

    file:close()

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
    local function mock_file(data)
        return {
            data = data,
            pointer = 1,
            read = function(self, size)
                local tmp = self.pointer
                self.pointer = self.pointer + size
                local readval = self.data.sub(self.data, tmp, self.pointer - 1)
                --print("i="..tmp..", j="..self.pointer..", size="..size
                --        ..", readval:len()="..readval:len())
                return readval
            end
        }
    end

    -- Test variable size fields.
    check_test(function()

        -- Test simple value 0.
        local value = 0
        local binary = value_to_var_len_encoding(value)
        assert_equals(1, binary:len())
        local size, value_recovered = read_var_len_value(mock_file(binary))
        assert_equals(binary:len(), size)
        assert_equals(value, value_recovered)

        -- Test values at borders of 7-bit bytes.
        for i = 1,3 do
            -- Do on low side of 7-bit number boundary.
            local value = bit.lshift(1, 7 * i) - 1
            local binary = value_to_var_len_encoding(value)
            assert_equals(i, binary:len())
            local size, value_recovered = read_var_len_value(mock_file(binary))
            assert_equals(binary:len(), size)
            assert_equals(value, value_recovered)

            -- Do on high side of 7-bit number boundary.
            local value = bit.lshift(1, 7 * i)
            local binary = value_to_var_len_encoding(value)
            assert_equals(i + 1, binary:len())
            local size, value_recovered = read_var_len_value(mock_file(binary))
            assert_equals(binary:len(), size)
            assert_equals(value, value_recovered)
        end

        -- Max value test.
        local value = bit.lshift(1, 7 * 4) - 1
        local binary = value_to_var_len_encoding(value)
        assert_equals(4, binary:len())
        local size, value_recovered = read_var_len_value(mock_file(binary))
        assert_equals(binary:len(), size)
        assert_equals(value, value_recovered)
    end)

    -- Test event parsers.
    check_test(function()

		-- Create a simple event.
        local event = {}
		event.delta_time = 1021
        event.command = 0xff
        event.meta_command = 0x2f
        event.data = 0
        print("test event:")
        print(event)

		-- Serialize the simple event.
        local binary = event_parser:tostring(event)
        local from_binary = {}
        event_parser:read(mock_file(binary), from_binary)
        print("from_binary:")
        print(from_binary)

		-- Check equality.
		for key, value in pairs(event) do
			assert_equals(value, from_binary[key])
		end

		-- Serialize and read again since we only simulated essential
		-- fields.
        local binary = event_parser:tostring(from_binary)
        local from_binary_again = {}
        event_parser:read(mock_file(binary), from_binary_again)

		-- Check equality.
		for key, value in pairs(from_binary) do
			assert_equals(value, from_binary_again[key])
		end

		-- See if we have the midi folder.
		local data_dir = "midi"
		local stat, _ = pcall(function() lfs.dir(data_dir) end)
		if stat then

			-- Get a random file from midi.
			mid_files = {}
			for filename in lfs.dir(data_dir) do
				if filename:lower():find(".mid") ~= nil then
					table.insert(mid_files, filename)
				end
			end
			mid_file = mid_files[math.random(1, #mid_files)]
			print("Using .mid file: "..mid_file)

			-- Read and tostring the file.
			data1 = mid.read(data_dir.."/"..mid_file)
			print(data1.num_tracks)
			for i, track in ipairs(data1.tracks) do
				print("Track "..i.." has "..#track.events.." events.")
			end

		end

    end)
end

return {
    HEADER = HEADER,
    TRACK_HEADER = TRACK_HEADER,
    MODE_SINGLE_TRACK = MODE_SINGLE_TRACK,
    MODE_MULTI_SYNCH = MODE_MULTI_SYNCH,
    MODE_MULTI_ASYNCH = MODE_MULTI_ASYNCH,
    read = read,
    write = write,
    _test = _test,
}

