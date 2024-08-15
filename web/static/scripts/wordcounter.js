document.getElementById("text-box").addEventListener("input", countWords);    function countWords()
    {
        let text=document.getElementById("text-box").value.trim();
        let textArray=[];
        if (text !== "")
        {
            let newText="";
            for (let i=0; i < text.length; i++)
            {
                if (text[i] === " " && i < text.length-1)
                {
                    if (text[i+1] !== " ")
                    {
                        newText+=text[i];
                    }
                }
                else if (text[i] === "\n" && i < text.length-1)
                {
                    if (text[i+1] !== "\n")
                    {
                        newText+=" ";
                    }
                }
                else
                {
                    newText+=text[i];
                }

            }
            textArray=newText.split(" ");
        }
        document.getElementById("word-count").innerHTML=`${textArray.length} Words`;
    }