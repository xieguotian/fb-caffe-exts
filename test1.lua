require 'fb.luaunit'
local t2c = require 'torch2caffe.lib'
require 'loadcaffe'
require 'nn'
require 'cunn'
require 'cudnn'
--require 'LocalNormalization'

-- local logging=  require 'fb.util.logging'
Test = {}
g_SLOW = "" -- set to "" to run slow tests

local fwk
if pcall(function() require 'cudnn' end) then
    print("Using `cudnn`")
    fwk = nn
else
    print("Using `nn`")
    fwk = nn
end

local function check(module, input_dims)
    module:apply(function(m) m:evaluate() end)
    local opts = {
            prototxt=os.tmpname(),
            caffemodel=os.tmpname(),
            inputs={{name="data", input_dims=input_dims}},
    }
    t2c.run(opts, module)
    return opts
end

local function check_opts(module, opts)
    module:apply(function(m) m:evaluate() end)
    opts.prototxt=os.tmpname()
    opts.caffemodel=os.tmpname()
    t2c.run(opts, module)
end
--[[
function Test:toy2()
    print(model.net)
    check(model.net,{1,3,200,66})
end
--]]
torch.setdefaulttensortype('torch.CudaTensor')
function Test:testToy()
    -- Hang Zhang form NVIDIA
    local net=nn.Sequential()
    local model=nn.Sequential()
    model:add(fwk.SpatialConvolution(3,24,5,5,2,2,0,0))
    model:add(fwk.SpatialBatchNormalization(24))
    model:add(nn.ELU())
    model:add(nn.SpatialDropout(0.25))
    model:add(fwk.SpatialConvolution(24,36,5,5,2,2,0,0))
    model:add(fwk.SpatialBatchNormalization(36))
    model:add(nn.ReLU())
    model:add(nn.SpatialDropout(0.25))
    model:add(fwk.SpatialConvolution(36,48,5,5,2,2,0,0))
    model:add(fwk.SpatialBatchNormalization(48))
    model:add(nn.ReLU())
    model:add(nn.SpatialDropout(0.25))
    model:add(fwk.SpatialConvolution(48,64,3,3,1,1,0,0))
    model:add(fwk.SpatialBatchNormalization(64))
    model:add(nn.ReLU())
    model:add(nn.SpatialDropout(0.25))
    model:add(fwk.SpatialConvolution(64,64,3,3,1,1,0,0))
    model:add(fwk.SpatialBatchNormalization(64))
    model:add(nn.ReLU())
    model:add(nn.SpatialDropout(0.25))
    model:add(nn.Reshape(64*18))
    model:add(nn.Linear(64*18,100))
    model:add(nn.ReLU())
    local model_= nn.Sequential()
    model_:add(nn.Linear(100,50))
    model_:add(nn.ReLU())
    model_:add(nn.Linear(50,10))
    model_:add(nn.ReLU())
    model_:add(nn.Linear(10,1))
    net:add(model)
    net:add(model_)
    net=nn.utils.recursiveType(net, 'torch.FloatTensor')
    check(net,{1,3,200,66})
    torch.save("pure_toy.t7b",net)
end

LuaUnit:main()
