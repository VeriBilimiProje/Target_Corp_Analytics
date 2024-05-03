# Centered title "data sapiens"
st.title("data sapi̇ens")

# Read the contents of the main.md file
with open("main.md", "r", encoding="utf-8") as file:
    markdown_text = file.read()

# Custom CSS to center align and make the background transparent
custom_css = """
        <style>
            .markdown-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 80vh;
                background-color: rgba(255, 255, 255, 0.5);
            }
        </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# Display the transparent and centered markdown content
st.markdown(markdown_text, unsafe_allow_html=True)


st.markdown('''
<div style="display: flex;">
    <a href="https://linktr.ee/mrakar" style="text-decoration: none; color: black; margin-right: 10px;">
        <div style="width: 75px; height: 75px; overflow: hidden; border-radius: 50%; padding: 10px; border: 2px solid black;">
            <img src="https://cdn.iconscout.com/icon/free/png-512/free-kaggle-3521526-2945029.png?f=webp&w=512" style="width: 100%; height: auto;">
        </div>
        <div style="text-align: center;">kaggle</div>
    </a>
    <a href="https://linktr.ee/mrakar" style="text-decoration: none; color: black; margin-right: 10px;">
        <div style="width: 75px; height: 75px; overflow: hidden; border-radius: 50%; padding: 10px; border: 2px solid black;">
            <img src="https://cdn.iconscout.com/icon/free/png-512/free-github-159-721954.png?f=webp&w=512" style="width: 100%; height: auto;">
        </div>
        <div style="text-align: center;">github</div>
    </a>
</div>
''', unsafe_allow_html=True)