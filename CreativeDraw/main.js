var model;
var classNames = [];
var canvas;
var coords = [];
var mousePressed = false;
var mode;
var selected;
var candidate = [];
var selectedNum;
var selectedNum = -2;
var str;
var lang;
var yesNum = 0;
var img;
var trans_kr = {};
var trans_en = {};
var trans_ch = {};
var trans_ja = {};
var kodata = [];
var endata = [];
var jadata = [];
var chdata = [];

LoadTrans();
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
        if(lang == 'kr'){
        sym.innerHTML = trans_kr[top5[i]]
        }
        else if(lang == 'en'){
        sym.innerHTML = trans_en[top5[i]]
        }
        else if(lang == 'jp'){
        sym.innerHTML = trans_ja[top5[i]]
        }
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
    if(lang=='kr'){document.getElementById('status').innerHTML = '그림을 그려주세요&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.AI가 그림을 다시 그려줍니다.';}
    if(lang=='en'){document.getElementById('status').innerHTML = 'Please draw a picture.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AI redraws the picture.';}
    if(lang=='ja'){document.getElementById('status').innerHTML = '絵を描いてください。&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AIが絵を描き直してくれます。';}
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
//사용자PC에 저장하는 기능은 완성함, 서버 주소입력하여 서버에 저장되는지 확인해야함
function save()
{
    var canvas = document.getElementById("canvas");
    var saveButton = document.getElementById("saveLink");

      saveButton.addEventListener("click", function () {
        // 캔버스의 이미지를 데이터 URL로 가져옴
        var dataURL = canvas.toDataURL("image/png");

        // 사용자의 PC에 이미지 저장
        var link = document.createElement("a");
        link.href = dataURL;
        link.download = "myImage.png";
        link.click();

        // 서버로 이미지를 전송하여 저장
        var serverURL = "서버 주소"; // 실제 서버 주소로 변경해야 함
        fetch(serverURL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: dataURL,
          }),
        })
          .then(function (response) {
            if (response.ok) {
              alert("그림이 성공적으로 서버에 저장되었습니다.");
            } else {
              alert("그림 저장에 실패했습니다.");
            }
          })
          .catch(function (error) {
            console.error("오류 발생:", error);
          });
      });
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
            img.style.position = "absolute";
            img.style.marginLeft = "620px";
            img.style.marginTop = "-960px";
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
            img.style.position = "absolute";
            img.style.marginLeft = "620px";
            img.style.marginTop = "-960px";
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
        var str;
        if (lang == 'kr')
            str = '<a href="mailto:pear1@ajou.ac.kr?subject=학습을 위한 데이터 제공">무엇을 그렸는지 메일로 알려주세요! </a>';
        else if (lang == 'en')
            str = '<a href="mailto:pear1@ajou.ac.kr?subject=Providing training data">Please mail to us and let us know what you drew</a>';
        else if (lang == 'jp')
            str = '<a href="mailto:pear1@ajou.ac.kr?subject=AIのための学習データの提供">描いたものをメールでお知らせください。</a>';
        element.innerHTML = str;
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
        str = '지금 그린 것이 ' + trans_kr[selected] + '가 맞나요?'
    else if (lang == 'en')
        str = 'Is the ' + selected + ' you drew right?' ;
    else if (lang == 'jp')
        str = '今描いたのは' + trans_ja[selected] + 'が正しいですか?'
    element.innerText = str;
}
//단어 번역

function LoadTrans() 
{    var ko = [];
     var en = [];
     var ja = [];
     var ch = [];
     readTextFile("translation_en.txt",'en');
     readTextFile("translation_ko.txt",'ko');
     readTextFile("translation_ja.txt",'ja');
     readTextFile("translation_zh-ch.txt",'ch');
     ko = kodata.split("\r\n");
     en = endata.split("\r\n");
     ja = jadata.split("\r\n");
     ch = chdata.split("\r\n");
 	for (var i=0; i<100; i++)
	{
        	trans_kr[en[i]] = ko[i];
		trans_ja[en[i]] = ja[i];
		trans_ch[en[i]] = ch[i];
		trans_en[en[i]] = en[i];
    	}
}


function readTextFile(file, ln)
{
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                var allText = rawFile.responseText;

                if(ln == 'ko')
		{
			kodata=allText;
                }
                else if(ln == 'en')
                {
			endata=allText;
                }
		else if(ln == 'ja')
                {
			jadata=allText;
                }
		else if(ln == 'ch')
                {
			chdata=allText;
                }

            }
        }
    }
    rawFile.send(null);
}
