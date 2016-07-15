require 'nn'
require 'cunn'
local pl = require 'pl.import_into'()
local py = require 'fb.python'
local torch = require 'torch'
local torch_layers = require 'torch2caffe.torch_layers'
local t2c = py.import('torch2caffe.lib_py')

-- Figure out the path of the model and load it
local path = arg[1]
local basename = paths.basename(path, 't7b')
local ext = path:match("^.+(%..+)$")
local model = nil
if ext == '.t7b' or ext == '.t7' then 
    torch_net = torch.load(path)
else
    assert(false, "We assume models end in either .t7b or .t7")
end

local function evaluate_caffe(caffe_net, inputs)
    local input_kwargs = {}
    for i=1,#inputs do
        local input_spec = inputs[i]
        input_kwargs[input_spec.name] = input_spec.tensor
    end
    local py_caffe_output = caffe_net.forward(py.kwargs, input_kwargs)
    local caffe_output_list = py.reval(t2c.format_output(py_caffe_output))
    local caffe_output_length = py.eval("len(a)", {a=caffe_output_list})
    local caffe_outputs = {}
    for i=0,caffe_output_length-1 do
        table.insert(caffe_outputs,
                     torch.FloatTensor(
                         torch.totable(py.eval(caffe_output_list[i]))))
    end
    return caffe_outputs
end


local function debug_nets(caffe_net, torch_net)
    py.reval(t2c.debug_net(caffe_net))
    torch_net:apply(
        function(m)
            if m.output then
                local sizes = {}
                local sums = {}
                if type(m.output) == 'table' then
                    for i=1,#m.output do
                        table.insert(sizes, m.output[i]:size())
                        table.insert(sums, torch.sum(m.output[i]))
                    end
                else
                    sizes = torch.totable(m.output:size())
                    sums = torch.sum(m.output)
                end
                print("Layer %s, %s, Sum: %s",
                              torch.typename(m),
                              sizes,
                              sums)
            end
        end
    )
end

local function inputs_to_torch_inputs(inputs, type)
    if #inputs == 1 then
        return inputs[1].tensor:type(type)
    end
    local tensors = {}
    for i=1,#inputs do
        table.insert(tensors, inputs[i].tensor:type(type))
    end
    return tensors
end


local function test()
    if  torch_net.net then
        torch_net=torch_net.net
    end
    torch_net:apply(function(m) m:evaluate() end)

    local opts = {}
    opts.prototxt = string.format('%s.prototxt', basename)
    opts.caffemodel = string.format('%s.caffemodel', basename)
    local caffe_net = t2c.load(opts)
-- 
    local inputs={}
    local tensor = torch.rand(table.unpack({1,3,224,224})):float()
    table.insert(inputs, {name='data', tensor=tensor})
    print ("\n\n\n\nTesting Caffe Model\n\n\n\n")
    
    local caffe_outputs = evaluate_caffe(caffe_net, inputs)
 
    local torch_outputs
    -- Some networks only accept CUDA input.
    local ok, err = pcall(function()
            torch_net:float()
            local torch_inputs = inputs_to_torch_inputs(
                inputs, 'torch.FloatTensor')
            torch_outputs = torch_net:forward(torch_inputs)
    end)
    if not ok then
        print("\n\n\nGot error running forward: %s", err)
        torch_net:cuda()
        local torch_inputs = inputs_to_torch_inputs(
            inputs, 'torch.CudaTensor')
        torch_outputs = torch_net:forward(torch_inputs)
        --error('not ok')
    end
    
    if type(torch_outputs) == "table" then
        for i=1,#torch_outputs do
            torch_outputs[i] = torch_outputs[i]:float()
        end
    else
        torch_outputs = {torch_outputs:float()}
    end
    
    if #caffe_outputs ~= #torch_outputs then
        error("Inconsistent output blobs: Caffe: %s, Torch: %s",
                       #caffe_outputs, #torch_outputs)
        error("Inconsistent output blobs")
    end

    for i = 1,#caffe_outputs do
        local torch_output = torch_outputs[i]
        local caffe_output = caffe_outputs[i]
        print("Caffe norm: %s, Torch norm: %s",
                      torch.norm(caffe_output), torch.norm(torch_output))
        if not caffe_output:isSameSizeAs(torch_output) then
            error("Inconsistent output size: Caffe: %s, Torch: %s",
                           caffe_output:size(), torch_output:size())
            error("Inconsistent output sizes")
        end

        local max_absolute_error = (caffe_output - torch_output):abs():max()
        print("Maximum difference between Caffe and Torch output: %s",
                      max_absolute_error)
        if 1 then --(max_absolute_error > 0.001) then
            debug_nets(caffe_net, torch_net)
            if os.getenv('LUA_DEBUG_ON_ERROR') then
                require('fb.debugger').enter()
            end
            if (max_absolute_error > 0.001) then  
                error("Error in conversion!")
            end
        end
    end    
    print (caffe_outputs[1])
end
--
test()

