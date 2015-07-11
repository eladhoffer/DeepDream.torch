require 'nn'
require 'cunn'
require 'cudnn'
require 'image'


local cuda = false
torch.setdefaulttensortype('torch.FloatTensor')
--net = torch.load('./GoogLeNet.t7')
net = torch.load('./OverFeatModel.t7'):float()
--net:training()


local Normalization = {mean = 118.380948/255, std = 61.896913/255}

function reduceNet(full_net,end_layer)
    local net = nn.Sequential()
    for l=1,end_layer do
        net:add(full_net:get(l))
    end
    return net
end

function make_step(net, img, end_layer,clip,step_size, jitter)
    local step_size = step_size or 0.01
    local jitter = jitter or 32
    local layer = layer or 15
    local clip = clip
    if clip == nil then clip = true end

    local ox = 0--2*jitter - math.random(jitter)
    local oy = 0--2*jitter - math.random(jitter)
    img = image.translate(img,ox,oy) -- apply jitter shift
    local dst, g
    if cuda then
        local cuda_img = img:cuda():view(1,img:size(1),img:size(2),img:size(3))
        dst = net:forward(cuda_img)
        g = net:updateGradInput(cuda_img,dst):float():squeeze()
    else
        dst = net:forward(img)
        g = net:updateGradInput(img,dst)
    end
    -- apply normalized ascent step to the input image
    img:add(g:mul(step_size/torch.abs(g):mean()))


    img = image.translate(img,-ox,-oy) -- apply jitter shift
    if clip then
        bias = Normalization.mean/Normalization.std
        img:clamp(-bias,1/Normalization.std-bias)
    end
    return img
end

function deepdream(net, base_img, iter_n, octave_n, octave_scale, end_layer, clip, visualize)

    local iter_n = iter_n or 10
    local octave_n = octave_n or 4
    local octave_scale = octave_scale or 1.4
    local end_layer = end_layer or 20
    local net = reduceNet(net, end_layer)
    local clip = clip
    if clip == nil then clip = true end
    -- prepare base images for all octaves
    local octaves = {}
    octaves[octave_n] = torch.add(base_img, -Normalization.mean):div(Normalization.std)
    local _,h,w = unpack(base_img:size():totable())

    for i=octave_n-1,1,-1 do
        octaves[i] = image.scale(octaves[i+1], math.ceil((1/octave_scale)*w), math.ceil((1/octave_scale)*h),'simple')
    end


    local detail
    local src

    for octave, octave_base in pairs(octaves) do
        src = octave_base
        local _,h1,w1 = unpack(src:size():totable())
        if octave > 1 then
            -- upscale details from the previous octave
            detail = image.scale(detail, w1, h1,'simple')
            src:add(detail)
        end
        for i=1,iter_n do
            src = make_step(net, src, end_layer, clip)
            if visualize then
                -- visualization
                vis = torch.mul(src, Normalization.std):add(Normalization.mean)

                if not clip then -- adjust image contrast if clipping is disabled
                    vis = vis:mul(1/vis:max())
                end

                image.display(vis)
            end
        end
        -- extract details produced on the current octave
        detail = src-octave_base
    end
    -- returning the resulting image
    src:mul(Normalization.std):add(Normalization.mean)
    return src

end

img = image.load('./sky1024px.jpg')
x = deepdream(net,img)
image.display(x)
