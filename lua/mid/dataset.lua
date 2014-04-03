--[
-- Tools to represent a .mid file as a Torch dataset.
--
-- (c) 2014 Brandon L. Reiss
--]
require "torch"
require "io"
require "math"

require "util"
require "test_util"
local mid_data = require "mid/data"
local mid_event = require "mid/event"

local CODES = mid_event.CODES
local event_parser = mid_event.parser

local binary_search = util.binary_search
local string_io = util.string_io

--- Number of unique note values allowed by MIDI.
local NOTE_DIMS = 128

local dataset = {}

local function raster_to_abs_time(t_raster, t_min, gcd)
    return t_min + ((t_raster - 1) * gcd)
end

local function abs_to_raster_time(t_abs, t_min, gcd)
    return 1 + ((t_abs - t_min) / gcd)
end

--- A simple clock converting raw clock time to raster time.
local function raster_clock(t_min, t_max, gcd)

    local clock = {}

    clock.min = t_min
    clock.max = t_max
    clock.gcd = gcd
    clock.abs_range = t_max - t_min
    clock.raster_range = clock.abs_range / gcd

    function clock.to_raster(time)
        return abs_to_raster_time(time, t_min, gcd)
    end

    function clock.from_raster(time)
        return raster_to_abs_time(time, t_min, gcd)
    end

    function clock.new_abs_time_end(time)
        return raster_clock(t_min, time, gcd)
    end

    function clock.new_raster_time_end(time)
        return raster_clock(t_min, clock.from_raster(time), gcd)
    end

    return clock
end

--- Compute the gcd of either
--    1) a pair of values (a, b) or
--    2) a list of values (a, b, c, ...)
local function gcd(a, b)

    -- Support either (a, b) or {a, b, c, ...}.
    local values
    if "table" == type(a) then
        values = a
    else
        values = {a, b}
    end
    if #values < 2 then
        print(values)
        error("Must have at least 2 values for gcd")
    end

    --- Helper for gcd using Euclid's algorithm.
    local function gcd_pair(a, b)
        if b == 0 then
            return a
        else
            return gcd_pair(b, a % b)
        end
    end

    -- The gcd of the list is the gcd of any subsequence of the list and the
    -- next element.
    local gcd = gcd_pair(values[1], values[2])
    for i = 2,(#values - 1) do
        gcd = gcd_pair(gcd, values[i])
    end

    return gcd
end

--- Create a hash code for a time signature event. We need to separate
-- learning machines by the timing of their music.
local function hash_time_signature_event(event)
    -- Only delta time of 0 supported.
    if 0 ~= event.delta_time then
        error("Time signature set at delta != 0")
    end

    return string.format(
            "%d/%d-%d-%d",
            event.numerator,
            event.denominator,
            event.n32_notes_per_qrtr,
            event.metro_ticks_per_click)
end

--- Create a discretization of the midi file such that each tensor column is a
-- simple accounting of the notes that are on and off separated by channel.
-- The notes are codes from 0-127.
--
-- The tensor for the entire piece will be
--     (channels * NOTE_DIMS) x raster_clock.raster_range
--
-- We require hashing events by note and then sorting them by time.  The
-- entire sequence length is known, so we create an empty bitmap and then
-- fill in the notes where they occur.
local function rasterize_song(channels, raster_clock, note_events)

    local channel_note_dims = #channels * NOTE_DIMS
    local rasterized = torch.zeros(channel_note_dims, raster_clock.raster_range)

    --print(note_events)
    --print(rasterized:size())

    for channel, notes in pairs(note_events) do

        local channel_idx = binary_search(channels, channel)
        local row_idx =  1 + ((channel_idx - 1) * NOTE_DIMS)
        local channel_data = rasterized:narrow(1, row_idx, NOTE_DIMS)
        --print("channel="..channel..", channel_idx="..channel_idx)

        --print(notes)
        for note, clock_events in pairs(notes) do

            local note_channel_data = channel_data:narrow(1, note, 1)
            --print(note_channel_data:size())

            local note_begin, velocity = nil, nil

            --print("#clock_events="..#clock_events)
            for _, clock_event in ipairs(clock_events) do

                local clock = clock_event.clock
                local event = clock_event.e

                if event.midi == CODES.midi.note_on then
                    note_begin = clock
                    velocity = event.velocity / 255.0

                elseif event.midi == CODES.midi.note_off
                    and note_begin ~= nil then
                    -- Set range for this note on with velocity.
                    local t_raster = raster_clock.to_raster(note_begin)
                    local duration = raster_clock.to_raster(clock) - t_raster
                    --print(clock_event.e.channel, clock_event.e.note_number,
                    --      note_begin, clock, duration)
                    note_channel_data:narrow(2, t_raster, duration):fill(velocity)
                    note_begin, velocity = nil, nil
                end

            end
        end
    end

    return rasterized
end

--- Scan midi data and recover note data.
--
-- Keys in return table:
--    raster_clock =>
--      a clock for converting between absolute and raster time
--    channel_to_track =>
--      a map of channel id to track index
--    time_sig =>
--      data time signature
--    note_events =>
--      a map of note events { channel -> { note_idx -> {events} } }
local function extract_note_event_data(middata)

    local note_events = {}

    -- Initialize other data that we are collecting.
    local channel_to_track, time_hash = {}, nil
    local min_clock, max_clock, clock_gcd = math.huge, -1, 0

    -- Loop over all tracks and events and collect clock stats for note
    -- events.
    for track_idx, track in ipairs(middata.tracks) do

        local clock = 0
        for event_idx, event in ipairs(track.events) do

            clock = clock + event.delta_time

            local note_on = event.midi == CODES.midi.note_on
            local note_off = event.midi == CODES.midi.note_off
            if note_on or note_off then

                -- Get channel-specific list of events for this note.
                local channel = event.channel
                channel_to_track[channel] = track_idx

                local notes = util.get_or_default(note_events, channel)
                local clock_events = util.get_or_default(notes, event.note_number)
                    
                -- Wrap events as a clock event with absolute time.
                table.insert(clock_events, { clock = clock, e = event })

                --print(channel, event_idx, clock, event.note_number, note_off, note_on)

                clock_gcd = gcd(clock_gcd, clock)
                min_clock = math.min(min_clock, clock)
                max_clock = math.max(max_clock, clock)

            elseif event.meta == CODES.meta.time_sig then
                time_hash = hash_time_signature_event(event)
            end
        end
    end

    local raster_clock = raster_clock(min_clock, max_clock, clock_gcd)
    --print(raster_clock)

    return {
        raster_clock = raster_clock,
        channel_to_track = channel_to_track,
        time_sig = time_hash.."-"..#channel_to_track.."-"..clock_gcd,
        events = note_events,
    }

end

--- Join data into a source.
local function make_source(filename, middata,
                           note_event_data, channel_order, rasterized)
    return {
        name = filename,
        data = rasterized,
        middata = middata,
        channel_to_track = note_event_data.channel_to_track,
        channel_order = channel_order,
        raster_clock = note_event_data.raster_clock,
        time_sig = note_event_data.time_sig
    }
end

--- Load a .mid file as a torch dataset.
--@param dir the directory containing .mid files
--@param time_sig the time signature in the form
--    num/denom-32notesperquarter-ticksperclick-channels-gcd
--@param input_len the number of discrete time steps per input point X
--@param target_len the number of discrete time steps per target point Y
--@param pct_train the proportion of points to use as training data
--@return train and test sets of pairs (X, Y)
function dataset.load(dir, time_sig, input_len, target_len, pct_train)

    -- Get all midi files in path.
    mid_files = {}
    for filename in lfs.dir(dir) do
        if filename:lower():find(".mid") ~= nil then
            table.insert(mid_files, filename)
        end
    end

    -- Collect rasterized data matching time signature.
    sources = {}

    -- For each file.
    for _,filename in ipairs(mid_files) do

        local file_path = dir.."/"..filename
        local file = io.open(file_path, "rb")

        local status, err = pcall(function()

            -- Load MIDI data.
            middata = mid_data.read(file)

            note_event_data = extract_note_event_data(middata)

            -- Get sorted list of channels from table of active channels.
            local channel_order = {}
            for channel in pairs(note_event_data.channel_to_track) do
                table.insert(channel_order, channel)
            end
            table.sort(channel_order)

            -- Check that time signature matches filter type.
            if note_event_data.time_sig == time_sig then

              -- Rasterize song and append to output.
              local rasterized = rasterize_song(
                      channel_order,
                      note_event_data.raster_clock, note_event_data.events)

              print("Adding file "..file_path)
              print("time sig: "..note_event_data.time_sig)
              print("dims: "..tostring(rasterized:size()))

              table.insert(sources,
                           make_source(filename, middata,
                                       note_event_data, channel_order, rasterized))
            end
        end)

        file:close()
    end

    -- For each valid source, split data into points and then partition into
    -- train and test sets. Each torch narrow() does not copy data so it can
    -- store efficiently overlapping subsequences of the source.
    local points = {}
    local input_target_len = input_len + target_len
    for _, source in pairs(sources) do
        local data = source.data
        for i = input_target_len,data:size()[2] do
            local X_begin = i - input_target_len + 1
            local Y_begin = i - target_len + 1
            local X, Y = data:narrow(2, X_begin, input_len),
                data:narrow(2, Y_begin, target_len)
            table.insert(points, {X, Y})
        end
    end

    -- Shuffle the points and then partition into train, test sets.
    util.shuffle(points)
    local num_train = math.ceil(#points * pct_train)
    local num_test = #points - num_train

    --- Train dataset.
    -- Must have size() and index operator [].
    local function data_train()
        local ds = {}

        ds.size = function()
            return num_train
        end

        local mt = getmetatable(ds)
        if not mt then
            mt = {}
        end
        mt.__index = function(ds, key)
            local num = tonumber(key)
            if num then
                if num <= num_train then
                    return points[num]
                else
                    return nil
                end
            else
                error("Must index by a number")
            end
        end
        setmetatable(ds, mt)

        return ds
    end

    --- Test dataset iterator.
    -- Must have size() and index operator [].
    local function data_test()
        local ds = {}

        ds.size = function()
            return num_test
        end

        local mt = getmetatable(ds)
        if not mt then
            mt = {}
        end
        mt.__index = function(ds, key)
            local num = tonumber(key)
            if num then
                if num > 0 then
                    return points[num_train + num]
                else
                    return nil
                end
            else
                error("Must index by a number")
            end
        end
        setmetatable(ds, mt)

        return ds
    end

    return {
        data_train = data_train,
        data_test = data_test,
        num_train = num_train,
        num_test = num_test,
        points = points,
        sources = sources,
    }
end

--- Add note events.
local function add_note_on(events, channel, note_number,
                           delta_time, velocity)
    table.insert(events, {
        midi = CODES.midi.note_on,
        command = bit.lshift(CODES.midi.note_on, 4) + channel,
        delta_time = delta_time,
        note_number =  note_number,
        velocity = velocity,
        common_name = "midi - note on",
        channel = channel,
    })
end

--- Add note events.
local function add_note_off(events, channel, note_number,
                            delta_time, velocity)
    table.insert(events, {
        midi = CODES.midi.note_off,
        command = bit.lshift(CODES.midi.note_on, 4) + channel,
        delta_time = delta_time,
        note_number =  note_number,
        velocity = 0,
        common_name = "midi - note off",
        channel = channel,
    })
end

--- Remove note events.
local copy_remove_note_events = function(events)
    local new_events = {}
    for _, event in pairs(events) do
        local is_midi = event.midi ~= nil
        local is_note = is_midi
                and (event.midi == CODES.midi.note_on
                        or event.midi == CODES.midi.note_off)
        if not is_midi or not is_note then
            table.insert(new_events, event)
        end
    end
    return new_events
end

--- Transform a rasterized song into midi data using a dataset source to
--specify base data such as tempos, instruments, and so fourth.
function dataset.compose(from_source, song, debounce_thresh)

    -- Ignore jumps of +/- debounce_thresh.
    debounce_thresh = debounce_thresh or 0

    local source = util.copy(from_source)
    source.data = song
    source.raster_clock = from_source.raster_clock.new_raster_time_end(song:size(2))
    local raster_clock = source.raster_clock

    source.middata = util.copy(source.middata)
    local middata = source.middata
    middata.tracks = util.copy(middata.tracks)
    local tracks = middata.tracks

    for raster_idx, channel in ipairs(source.channel_order) do

        local chan_row_begin = 1 + ((raster_idx - 1) * NOTE_DIMS)
        local channel_data = song:narrow(1, chan_row_begin, NOTE_DIMS)

        --print('channel_data:size() = '..tostring(channel_data:size()))

        local track_idx = source.channel_to_track[channel]
        tracks[track_idx] = util.copy(tracks[track_idx])
        local track = tracks[track_idx]

        -- For each event in the input song, create an event in the output.

        local on = {}
        track.events = copy_remove_note_events(track.events)
        local events = track.events
        local track_end_event = events[#events]
        events[#events] = nil

        local last_clock = 0
        for t_raster = 1, channel_data:size(2) do
            
            local notes_data = channel_data:narrow(2, t_raster, 1)
            local clock = raster_clock.from_raster(t_raster)

            --print('notes_data = '..tostring(notes_data))
            --print('notes_data:size() = '..tostring(notes_data:size()))

            -- Go over each note in the channel and see if we have an event.
            for note_idx = 1, NOTE_DIMS do

                -- Extract note velocity enforcing data bounds.
                local note_value_raw = notes_data[note_idx][1]

                local velocity
                if util.isnan(note_value_raw) then
                    velocity = 0
                else
                    velocity = math.max(0, math.min(255,
                            math.floor(note_value_raw * 255)))
                end

                --print(channel, note_idx, velocity)

                -- Assemble possible states for this note.

                local note_on = velocity > 0
                local note_on_already = on[note_idx] ~= nil

                local note_changing = note_on
                        and note_on_already
                        and (on[note_idx][2] < velocity - debounce_thresh
                            or on[note_idx][2] > velocity + debounce_thresh)

                local note_starting = not note_on_already and note_on
                local note_ending = note_on_already and not note_on

                -- Update state for this note.

                if note_ending or note_changing then
                    on[note_idx] = nil

                    local delta_time = clock - last_clock
                    add_note_off(events, channel, note_idx, delta_time)
                    last_clock = clock
                end

                if note_starting or note_changing then
                    on[note_idx] = { clock, velocity }

                    local delta_time = clock - last_clock
                    add_note_on(events, channel, note_idx, delta_time, velocity)
                    last_clock = clock
                end

            end

        end

        -- End all notes still on.
        local end_clock = raster_clock.from_raster(channel_data:size(2) + 1)
        for note_idx in pairs(on) do
            local delta_time = end_clock - last_clock
            add_note_off(events, channel, note_idx, delta_time)
            last_clock = end_clock
        end

        -- Update end-of-track event and append.
        track_end_event.delta_time = end_clock - last_clock
        table.insert(events, track_end_event)

        --print("track_idx="..track_idx..", #events="..#events..", chan_row_begin="..chan_row_begin)

        -- Determine track size.
        local str_file = string_io()
        for _, event in ipairs(events) do
            event_parser:write(event, str_file)
        end
        track.size = #str_file.data

    end

    return source
end

function dataset._test()

    local assert_equals = test_util.assert_equals
    local check_test = test_util.check_test

    -- Test raster and back.
    check_test(function()

        local MAX_TO_TEST = 10

        -- See if we have the midi folder.
        local data_dir = "../midi"
        local stat, _ = pcall(function() lfs.dir(data_dir) end)
        if stat then

            mid_files = {}
            for filename in lfs.dir(data_dir) do
                if filename:lower():find(".mid") ~= nil then
                    table.insert(mid_files, filename)
                end
            end

            util.shuffle(mid_files)
            local num_to_test = 1 --math.min(MAX_TO_TEST, #mid_files)

            for i = 1, num_to_test do
                local filename = mid_files[i]
                --local file_path = data_dir.."/"..filename
                local file_path = "../midi/027800b_.mid"
                print("Using .mid file: "..file_path)

                local middata = mid_data.read(io.open(file_path))

                -- Keep 10 events.
                for i, track in ipairs(middata.tracks) do
                    local events_new = {}
                    num_events = math.min(8, #track.events - 1)
                    for j = 1, num_events do
                        table.insert(events_new, track.events[j])
                    end
                    table.insert(events_new, track.events[#track.events])
                    middata.tracks[i].events = events_new
                end
                --print(middata)

                -- Rasterize and then write out.
                local note_event_data = extract_note_event_data(middata)

                -- Get sorted list of channels from table of active channels.
                local channel_order = {}
                for channel in pairs(note_event_data.channel_to_track) do
                    table.insert(channel_order, channel)
                end
                table.sort(channel_order)

                -- Rasterize song and append to output.
                local rasterized = rasterize_song(
                        channel_order,
                        note_event_data.raster_clock, note_event_data.events)

                -- Compose into a new source that should receive the same data.

                local source =
                        make_source(filename, middata,
                                    note_event_data, channel_order, rasterized)
                
                local new_source = dataset.compose(source, rasterized)
                print(new_source.channel_order)
                print(new_source.channel_to_track)

                print("Summary")
                for i = 1, #middata.tracks do
                    print(i, #source.middata.tracks[i].events,
                             #new_source.middata.tracks[i].events)
                    --for j = 1, #middata.tracks[i].events do
                    --    print(tostring(middata.tracks[i].events[j]),
                    --          tostring(new_source.middata.tracks[i].events[j]))
                    --end
                end
                --print(new_source.middata.tracks[5])
                --print(source.middata.tracks[5])

                -- TODO: How can the same note stop/start at the same time?

                --local mock_file = string_io()
                --mid_data.write(new_source.middata, mock_file)

                ---- Check that the written file matches exactly the source.
                --local mid_file_bin = io.open(file_path):read(1024*1024*128)
                --assert_equals(mid_file_bin, mock_file.data)

            end

        else
            error(data_dir.." not found")
        end

    end)

end

return dataset
