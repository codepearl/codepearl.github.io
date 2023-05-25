var model;
var canvas;
var classNames = [];
var canvas;
var coords = [];
var mousePressed = false;
var mode;
var selected;
var selected = -2;
var candidate = [];
var selectedNum;
var str;
var lang;
var yesNum = 0;
var img;

function getRandomNumber(min, max) 
{
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

$(function()
{
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 0;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 10;
    canvas.renderAll();

    canvas.on('mouse:up', function(e)
    {
        getFrame();
        mousePressed = false
    });
    canvas.on('mouse:down', function(e)
    {
        mousePressed = true
    });
    canvas.on('mouse:move', function(e)
    {
        recordCoor(e)
    });
})

function setTable(top5, probs)
{
    candidate = top5;
    selectedNum = -1;
    check();
    for (var i = 0; i < top5.length; i++)
    {
        let sym = document.getElementById('sym' + (i + 1))
        let prob = document.getElementById('prob' + (i + 1))
        sym.innerHTML = top5[i]
        prob.innerHTML = Math.round(probs[i] * 100)
    }
    createPie(".pieID.legend", ".pieID.pie");
}

function recordCoor(event)
{
    var pointer = canvas.getPointer(event.e);
    var posX = pointer.x;
    var posY = pointer.y;

    if (posX >= 0 && posY >= 0 && mousePressed)
    {
        coords.push(pointer)
    }
}

function getMinBox() {
    var coorX = coords.map(function(p) {
        return p.x
    });
    var coorY = coords.map(function(p) {
        return p.y
    });

    var min_coords = {
        x: Math.min.apply(null, coorX),
        y: Math.min.apply(null, coorY)
    }
    var max_coords = {
        x: Math.max.apply(null, coorX),
        y: Math.max.apply(null, coorY)
    }

    return {
        min: min_coords,
        max: max_coords
    }
}


function getImageData()
{
    const mbb = getMinBox()

    const dpi = window.devicePixelRatio
    const imgData = canvas.contextContainer.getImageData(mbb.min.x * dpi, mbb.min.y * dpi,
                                                      (mbb.max.x - mbb.min.x) * dpi, (mbb.max.y - mbb.min.y) * dpi);
    return imgData
}

function getFrame()
{
    if (coords.length >= 2)
    {
        //canvas에서 이미지 데이터 가져오기
        const imgData = getImageData()

        console.log("getFrame()");
        console.log(model.summary());
        //모델 예측하기
        const pred = model.predict(preprocess(imgData)).dataSync()

        const indices = findIndicesOfMax(pred, 5)
        const probs = findTopValues(pred, 5)
        const names = getClassNames(indices)

        setTable(names, probs)
    }

}

function getClassNames(indices)
{
    var outp = []
    for (var i = 0; i < indices.length; i++)
        outp[i] = classNames[indices[i]]
    return outp
}

//class name 불러오기
async function loadDict()
{
    loc = 'model/class_names.txt'
    
    await $.ajax(
    {
        url: loc,
        dataType: 'text',
    }).done(success);
}

function success(data)
{
    const lst = data.split(/\n/)
    for (var i = 0; i < lst.length - 1; i++)
    {
        let symbol = lst[i]
        classNames[i] = symbol
    }
}

function findIndicesOfMax(inp, count)
{
    var outp = [];
    for (var i = 0; i < inp.length; i++)
    {
        outp.push(i);
        if (outp.length > count)
        {
            outp.sort(function(a, b)
            {
                return inp[b] - inp[a];
            });
            outp.pop();
        }
    }
    return outp;
}

function findTopValues(inp, count)
{
    var outp = [];
    let indices = findIndicesOfMax(inp, count)
    for (var i = 0; i < indices.length; i++)
        outp[i] = inp[indices[i]]
    return outp
}

function preprocess(imgData)
{
    return tf.tidy(() =>
    {
        let tensor = tf.browser.fromPixels(imgData, numChannels = 1)

        const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat()

        const offset = tf.scalar(255.0);
        const normalized = tf.scalar(1.0).sub(resized.div(offset));

        const batched = normalized.expandDims(0)
        return batched
    })
}

async function start(cur_mode)
{
    mode = cur_mode
    model = await tf.loadLayersModel('model/model.json')
    model.predict(tf.zeros([1, 28, 28, 1])) 
    allowDrawing()
    await loadDict()
    console.log(model.summary());
}

function allowDrawing()
{
    canvas.isDrawingMode = 1;
    document.getElementById('status').innerHTML = 'Model Loaded';
    $('button').prop('disabled', false);
    var slider = document.getElementById('myRange');
    slider.oninput = function()
    {
        canvas.freeDrawingBrush.width = this.value;
    };
}

function erase()
{
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    coords = [];
}

function save(){    
    console.log('export image');
     if (!fabric.Canvas.supports('toDataURL')) {
      alert('This browser doesn\'t provide means to serialize canvas to an image');
    }
    else {
      window.open(canvas.toDataURL('png'));
    }
    }

function yes()
{
    if (selectedNum != -2)
    {
        if (yesNum == 0)
        {
            console.log('change image');
            console.log(selected);
            img = document.createElement("img");
            var src = 'img/' + selected + '_' + getRandomNumber(0,2) + '.jpg';
            img.src = src;
            img.width = 523;
            img.height = 475;
            img.alt = selected;
            document.body.appendChild(img);
            yesNum ++;
        }
        else
        {
            document.body.removeChild(img);
            console.log('change image');
            console.log(selected);
            img = document.createElement("img");
            var src = 'img/' + selected + '_' + getRandomNumber(0,2) + '.jpg';
            img.src = src;
            img.width = 523;
            img.height = 475;
            img.alt = selected;
            document.body.appendChild(img);
            yesNum ++;
        }
    }

}

function no()
{
    if (selectedNum >= 4)
    {
        console.log('submit to server');
        const element = document.getElementById('target');
        var str = '발전을 위해서 무엇을 그렸는지 알려주세요';
        element.innerText = str;
        //keyword를 입력받는게 필요함
        var keyword = 'something';
        var src = 'submit/' + keyword + '.png';
        fabric.Canvas.supports(src);
    }
    else check();
}

function check()
{
    selectedNum = selectedNum + 1;
    selected = candidate[selectedNum];
    console.log(selected);
    changeLanguage();
}

function changeLanguage()
{
    var radios = document.getElementsByName("language");

    for (var i = 0; i < radios.length; i++)
    {       
        if (radios[i].checked)
        {
            lang = (radios[i].value);
            console.log(lang);
            i = radios.length;
        }
    }

    const element = document.getElementById('target');
    var str;
    if (lang == 'kr')
        str = '지금 그린 것이 ' + selected + '가 맞나요?'
    else if (lang == 'en')
        str = 'Is the ' + selected + ' you drew right?' ;
    else if (lang == 'jp')
        str = '今描いたのは' + selected + 'が正しいですか?'
    element.innerText = str;
}