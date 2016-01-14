require 'nn'
require 'cutorch'
require 'warp_ctc'

function os.capture(cmd, raw)
    local f = assert(io.popen(cmd, 'r'))
    local s = assert(f:read('*a'))
    f:close()
    if raw then return s end
    s = string.gsub(s, '^%s+', '')
    s = string.gsub(s, '%s+$', '')
    s = string.gsub(s, '[\n\r]+', ' ')
    return s
end

function reduce(list)
    local acc
    for k, v in ipairs(list) do
        if 1 == k then
            acc = v
        else
            acc = acc +  v
        end
    end
    return acc
end

function simpleTest()
    local cpu_acts = torch.Tensor({{0.1, 0.6, 0.1, 0.1,0.1},{0.1, 0.1, 0.6, 0.1, 0.1}}):float()
    local cpu_probs = nn.SoftMax():updateOutput(cpu_acts:double()):float()
    local cpu_grads = cpu_probs:clone():zero()

    local labels = {{1,2}}
    --local label_lengths = torch.Tensor({2}):int()

    local sizes = {2}
    --print(cpu_probs, cpu_grads, labels, sizes)

    print("CPU_cost:", reduce(cpu_ctc(cpu_acts, cpu_grads, labels, sizes)))
    print(cpu_grads)

    local cpu_grads = torch.Tensor():float()
    print("CPU_cost: score forward", reduce(cpu_ctc(cpu_acts, cpu_grads, labels, sizes)))

    local acts = cpu_acts:cuda()
    local grads = acts:clone():zero()

    --print(probs, grads, labels, label_lengths, sizes)

    local cost = reduce(gpu_ctc(acts, grads, labels, sizes))
    print("GPU_cost:", cost)
    print(grads)

    local grads = torch.Tensor():cuda()
    print("GPU_cost: score forward", reduce(gpu_ctc(acts, grads, labels, sizes)))

end

function mediumTest(multiplier)
    local cpu_acts = torch.Tensor({{0.1, 0.6, 0.1, 0.1,0.1},{0.1, 0.1, 0.6, 0.1, 0.1},
        {0.6, 0.1, 0.1, 0.1,0.1},{0.1, 0.1, 0.5, 0.2, 0.1}}):float()*multiplier
    local cpu_grads = cpu_acts:clone():zero()

    local labels = {{1,2},{1,2}}
    local sizes = {2, 2 }

    --print(cpu_probs, cpu_grads, labels, sizes)

    print("CPU_cost:", reduce(cpu_ctc(cpu_acts, cpu_grads, labels, sizes)))
    print(cpu_grads)

    local cpu_grads = torch.Tensor():float()
    print("CPU_cost: score forward", reduce(cpu_ctc(cpu_acts, cpu_grads, labels, sizes)))

    local acts = cpu_acts:cuda()
    local grads = acts:clone():zero()

    --print(probs, grads, labels, sizes)

    local cost = reduce(gpu_ctc(acts, grads, labels, sizes))
    print("GPU_cost:", cost)
    print(grads)

    local grads = torch.Tensor():cuda()
    print("GPU_cost: score forward", reduce(gpu_ctc(acts, grads, labels, sizes)))

end

function emptyLabelTest()
    local cpu_acts = torch.Tensor({{0.1, 0.6, 0.1, 0.1,0.1},{0.1, 0.1, 0.6, 0.1, 0.1},
        {0.6, 0.1, 0.1, 0.1,0.1},{0.1, 0.1, 0.5, 0.2, 0.1}}):float()
    local cpu_grads = cpu_acts:clone():zero()

    local labels = {{1,2},{}}
    local sizes = {2, 2 }

    --print(cpu_probs, cpu_grads, labels, sizes)

    print("CPU_cost:", reduce(cpu_ctc(cpu_acts, cpu_grads, labels, sizes)))
    print(cpu_grads)

    local cpu_grads = torch.Tensor():float()
    print("CPU_cost: score forward", reduce(cpu_ctc(cpu_acts, cpu_grads, labels, sizes)))

    local acts = cpu_acts:cuda()
    local grads = acts:clone():zero()

    --print(probs, grads, labels, sizes)

    local cost = reduce(gpu_ctc(acts, grads, labels, sizes))
    print("GPU_cost:", cost)
    print(grads)

    local grads = torch.Tensor():cuda()
    print("GPU_cost: score forward", reduce(gpu_ctc(acts, grads, labels, sizes)))

end

function getTargets()
    local outdim = 29 --TODO count chars.txt
    local file = io.open("data/sizes.txt", "r");

    if not file then
        print("File not found data/sizes.txt are you runnng the test from the tests dir?")
    end

    local sizes = {}
    for line in file:lines() do
        table.insert (sizes, tonumber(line));
    end


    local label_file = io.open("data/labels.txt", "r");
    local labels = {}

    for line in label_file:lines() do
        local current_labels = {}
        for w in line:gmatch("%S+") do
            table.insert (current_labels, tonumber(w));
        end
        table.insert (labels, current_labels);
    end

    return outdim, sizes, labels
end


function bigTest(minibatch_size)

    detected_OS = os.capture('uname', false)

    local outdim, raw_sizes, raw_labels = getTargets()

    -- truncate tables to given minibatch_size
    local sizes = {}
    local labels = {}

    local max_length = 0

    for idx = 1,minibatch_size do
        if raw_sizes[idx] > max_length then
            max_length = raw_sizes[idx]
        end

        table.insert(sizes, raw_sizes[idx])
        table.insert(labels, raw_labels[idx])
    end

    local minibatch_size = table.getn(sizes)

    print("Using minibatch size: ", #sizes)
    print("Using outdim size: ", outdim)
    print("Max size: ", max_length)

    torch.manualSeed(123)

    local cpu_acts = torch.rand(minibatch_size*max_length, outdim):float()
    local cpu_grads = cpu_acts:clone():fill(0)

    print("CPU_cost:", reduce(cpu_ctc(cpu_acts, cpu_grads, labels, sizes)))

    if detected_OS == "Darwin" then
        if cpu_grads:ne(cpu_grads):sum() > 0 then
            print(sys.COLORS.red .. ' cpu_grads after update has NaN/s')
        else
            print('cpu_grads do not have nans')
        end
    end

    local cpu_null_grads = torch.Tensor():float()
    print("CPU_cost: score forward", reduce(cpu_ctc(cpu_acts, cpu_null_grads, labels, sizes)))

    local acts = cpu_acts:cuda()
    local grads = acts:clone():zero()

    --print(probs, grads, labels, sizes)

    local cost = reduce(gpu_ctc(acts, grads, labels, sizes))
    print("GPU_cost:", cost)

    if detected_OS == "Darwin" then
        if grads:ne(grads):sum() > 0 then
            print(sys.COLORS.red .. ' gpu_grads after update has NaN/s')
        else
            print('gpu_grads do not have nans')
        end

        print("L2 norm grad diff: ", torch.norm(cpu_grads - grads:float()))

    end

    local grads = torch.Tensor():cuda()
    print("GPU_cost: score forward", reduce(gpu_ctc(acts, grads, labels, sizes)))

end

simpleTest()
mediumTest(1.0)
print("Stability test")
mediumTest(200.0) -- test SM stability if compiled with USE_NSM this will not have nans
print("Empty label test")
emptyLabelTest()
bigTest(32)
bigTest(64)
bigTest(96)
bigTest(111)