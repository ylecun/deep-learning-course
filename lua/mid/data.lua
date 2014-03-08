--[
-- Function to manipulate MID data.
--
-- (c) 2014 Brandon L. Reiss
--]

require "lfs"

require "util"
require "test_util"
require "mid/event"

local get_len_check_expected = event.get_len_check_expected

local string_to_int = util.string_to_int
local int_to_string = util.int_to_string
local string_io = util.string_io

local event_parser = event.parser

local data = {
    -- The MIDI header magic numbers.
    FILE_HEADER = "MThd",
    TRACK_HEADER = "MTrk",
    -- Track modes.
    MODE_SINGLE_TRACK = 0,
    MODE_MULTI_SYNCH = 1,
    MODE_MULTI_ASYNCH = 2,
}
local FILE_HEADER, TRACK_HEADER = data.FILE_HEADER, data.TRACK_HEADER

--- Open a .mid file.
function data.read(file)

    -- Read .mid header.
    local header = file:read(4)
    if header ~= FILE_HEADER then
        error({msg=string.format("MIDI file header(%s) != expected %s",
                header, FILE_HEADER)})
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
        -- Get track header magic number.
        local track_header = file:read(4)
        if track_header == nil then
            return data
        elseif track_header ~= TRACK_HEADER then
            error({msg=string.format("MIDI track header 0x%x != expected 0x%x",
                    track_header, TRACK_HEADER)})
        end
        track = {
            size = string_to_int(file:read(4)),
        }
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

end

--- Write a .mid file. Note that you may mock the file parameter using a
--string_io() and then recover the data from string_io.data.
function data.write(data, file)

    --- Check that result is true or else throw error(msg).
    local function check_result(result, msg)
        if result ~= true then
            error(msg)
        end
    end

    -- Write header.
    check_result(file:write(FILE_HEADER
            ..int_to_string(6, 4)
            ..int_to_string(data.track_mode, 2)
            ..int_to_string(data.num_tracks, 2)
            ..int_to_string(data.ticks_per_qrtr, 2)),
            "Failed writing header")

    -- Write tracks.
    for _,track in ipairs(data.tracks) do

        -- Write track header and size.
        check_result(file:write(TRACK_HEADER
                ..int_to_string(track.size, 4)),
                "Failed writing track header and size")

        -- Write events.
        for _,event in ipairs(track.events) do
            check_result(event_parser:write(event, file),
                    "Failed writing event")
        end

    end

end

--- Filter track events based on type codes.
--  Event filter codes are of the form:
--      system = {code, ...}
--      ctype = {code, ...}
--      meta = {code, ...}
--  See CODES for available types.
function data.filter_track(track, system, ctype, meta)

    -- Sort filtered types.
    local system = system or {}
    local ctype = ctype or {}
    local meta = meta or {}
    table.sort(system)
    table.sort(ctype)
    table.sort(meta)

    local filtered_track = {}
    for k,v in pairs(track) do
        filtered_track[k] = v
    end
    local size = track.size
    local filtered_events = {}

    for _, event in pairs(track.events) do
        -- Determine command type and then check filters.
        local add_event
        if event.ctype ~= nil then
            add_event = binary_search(ctype, event.ctype)
        elseif event.meta ~= nil then
            add_event = binary_search(meta, event.meta) > 0
        else
            add_event = binary_search(system, event.command) > 0
        end
        if add_event then
            table.insert(filtered_events, event)
        else
            local string_io = string_io()
            event_parser:write(event, string_io)
            size = size - string.len(string_io.data)
        end
    end

    filtered_track.events = filtered_events
    filtered_track.size = size
    return filtered_track
end

--- Run tests.
function data._test()

    local assert_equals = test_util.assert_equals
    local check_test = test_util.check_test

    -- Test read() and write().
    check_test(function()

        -- See if we have the midi folder.
        local data_dir = "../midi"
        local stat, _ = pcall(function() lfs.dir(data_dir) end)
        if stat then

            -- Get a random file from midi.
            mid_files = {}
            for filename in lfs.dir(data_dir) do
                if filename:lower():find(".mid") ~= nil then
                    table.insert(mid_files, filename)
                end
            end

            for _,filename in ipairs(mid_files) do
                local file_path = data_dir.."/"..filename
                print("Using .mid file: "..file_path)

                -- Read and write the file.
                local middata = data.read(io.open(file_path))
                for i, track in ipairs(middata.tracks) do
                    print("Track "..i.." has "..#track.events.." events.")
                end
                local mock_file = string_io()
                data.write(middata, mock_file)

                -- Check that the written file matches exactly the source.
                local mid_file_bin = io.open(file_path):read(1024*1024*128)
                assert_equals(mid_file_bin, mock_file.data)
            end

        else
            error(data_dir.." not found")
        end

    end)
end

return data

