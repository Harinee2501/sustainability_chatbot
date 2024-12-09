import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import ssl
import streamlit as st
from news_api import fetch_news
from research_fetcher import fetch_research, fetch_wikipedia_summary

# Fix SSL certificate issues for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define intents
intents = {
    "greetings": {
        "patterns": ["Hi", "Hello", "Hey", "Hi there", "Good morning", "Good evening", "What's up?", "Hey there", "Greetings"],
        "responses": [
            "Hello! I'm here to help with all things sustainability. How can I assist you today? Whether it's sustainable living tips or eco-friendly project ideas, feel free to ask.",
            "Hi there! Whether you're looking to reduce your carbon footprint or want advice on going green, I'm here to help you take steps towards a more sustainable lifestyle. How can I support you today?",
            "Hey! Excited to talk sustainability with you. Whether you're interested in eco-friendly products, renewable energy, or how to reduce waste, I'm all ears. What would you like to know?",
            "Good morning! Let's talk about making the world a better place. How can I help you with sustainability today? Feel free to ask any questions you have on green living, climate change, or eco-friendly habits."
        ]
    },
    "project_ideas": {
        "patterns": ["Can you suggest some projects?", "What are some SDG-related project ideas?", "Help me with sustainable project ideas", "Give me some project ideas for sustainability", "What are some cool environmental projects?"],
        "responses": [
            "**No Poverty**: Build an app that connects underserved communities with job opportunities, online education, and mental health resources. You could partner with NGOs or local businesses to provide remote training programs, online job fairs, and wellness services.",
            "**Zero Hunger**: Create a platform that connects people who have excess food with local food banks or shelters. The app could allow users to donate unused groceries, food from events, or restaurant leftovers to those in need, minimizing food waste and hunger.",
            "**Good Health and Well-being**: Develop an app that provides real-time air quality monitoring and health advice. This could include suggestions on how to stay healthy when pollution levels are high, such as using air purifiers or adjusting outdoor activities to avoid exposure to harmful air particles.",
            "**Quality Education**: Create a website or app that offers free online courses about sustainability. You could include interactive modules on topics like renewable energy, waste management, eco-friendly lifestyles, and sustainable agriculture, all tailored to different age groups and educational levels.",
            "**Gender Equality**: Start a project that helps women enter renewable energy careers by creating mentorship programs, online courses, and job boards. You could partner with clean energy companies to provide opportunities for women to upskill and network in this growing industry.",
            "**Decent Work and Economic Growth**: Create an app that tracks and rewards sustainable practices in local businesses. It could offer businesses incentives like recognition, certifications, or discounts on green supplies for adopting environmentally friendly practices such as energy efficiency, waste reduction, and sustainable sourcing.",
            "**Industry, Innovation, and Infrastructure**: Develop an eco-friendly building material made from recycled plastics or agricultural waste. This could be used in construction projects to reduce waste while providing affordable building materials for low-income communities.",
            "**Reduced Inequality**: Build a platform that connects people with disabilities to accessible, green jobs in sustainability sectors. The app could help with job search, offer specialized skill-building courses, and provide resources on how to make workplaces more inclusive and sustainable.",
            "**Responsible Consumption and Production**: Launch a mobile app that helps users track the sustainability of the products they purchase. It could provide information on carbon footprints, materials used, and environmental impact ratings to encourage more eco-conscious consumption choices.",
            "**Climate Action**: Create a gamified platform that encourages individuals and organizations to take climate-positive actions. The app could offer carbon offset credits, track emissions reductions, and offer rewards for actions such as planting trees, reducing water usage, or switching to renewable energy sources."
        ]
    },
    "recycling": {
        "patterns": ["How to recycle plastic?", "What can I do with glass?", "Where to dispose e-waste?", "How to recycle paper?", "What is the best way to recycle metal?", "How do I recycle clothes?", "How can I recycle electronics?"],
        "responses": [
            "Plastic: Always check local recycling guidelines to ensure you're recycling correctly. Some plastics can't be recycled due to contamination, so it's important to rinse them first. You can also explore upcycling projects to turn plastic waste into something useful, like art or home decor.",
            "Glass: Glass is one of the most recyclable materials. It can be recycled indefinitely without losing quality. Separate glass from other materials, and avoid mixing it with ceramics or mirrors, as these cannot be recycled with glass. Many municipalities offer curbside glass recycling programs.",
            "E-waste: Electronic waste should be taken to certified e-waste recycling centers. These facilities specialize in safely recycling computers, phones, and other electronics, extracting valuable metals like gold and copper, while preventing hazardous materials like lead and mercury from polluting the environment.",
            "Paper: Paper is one of the easiest materials to recycle, but it needs to be clean and dry. Avoid recycling paper with food residue, grease, or chemicals, as it can contaminate the recycling stream. You can also reuse paper for crafts, gift wrapping, or creating compost for gardens.",
            "Metal: Metals like aluminum and steel are highly recyclable. Aluminum cans, for example, can be melted down and reused to make new cans or even cars. It's important to remove any food waste from cans and clean them before recycling. You can also consider taking scrap metal to local recycling centers.",
            "Clothes: Donate gently used clothing to second-hand stores or organizations that provide clothes to those in need. If clothes are no longer wearable, look for textile recycling programs that can repurpose fabrics into insulation or other products. Avoid sending textiles to landfills whenever possible."
        ]
    },
    "carbon_footprint": {
        "patterns": ["How to calculate carbon footprint?", "What is my carbon footprint?", "Ways to reduce carbon footprint?", "How to minimize carbon emissions?", "What can I do to reduce my carbon footprint?", "How can I measure my carbon footprint?"],
        "responses": [
            "To calculate your carbon footprint, you can use an online calculator where you input information about your transportation (car, public transport, flight), energy usage (electricity, heating, cooling), diet (meat vs plant-based foods), and waste habits. The calculator will provide an estimate of your total emissions.",
            "Some ways to reduce your carbon footprint include driving less by walking or biking, using public transportation, eating a plant-based diet, reducing your use of single-use plastics, and choosing energy-efficient appliances. Switching to renewable energy sources, like solar or wind, can also make a big impact.",
            "Consider adopting a zero-waste lifestyle, where you reduce your consumption of disposable products and focus on buying in bulk or reusing items. Installing solar panels, switching to energy-efficient lighting, and using smart thermostats are all effective strategies for reducing energy consumption.",
            "Carbon offset programs allow you to invest in projects that reduce emissions elsewhere, such as reforestation, renewable energy projects, or methane capture. This can help balance out the emissions from your activities, especially in areas where direct reductions are not possible."
        ]
    },
    "renewable_energy": {
        "patterns": ["What are renewable energy sources?", "Explain solar energy", "What is wind energy?", "Tell me about hydropower.", "How does geothermal energy work?", "Why is renewable energy important?"],
        "responses": [
            "Renewable energy sources are energy sources that are replenished naturally and are sustainable in the long term. These include solar, wind, hydropower, geothermal, and biomass energy. They help reduce our reliance on fossil fuels, lower greenhouse gas emissions, and mitigate climate change.",
            "Solar energy harnesses the power of sunlight to generate electricity or heat. Photovoltaic (PV) panels convert sunlight directly into electricity, while solar thermal systems use mirrors to focus sunlight and generate heat for water or space heating. Solar energy is one of the most popular and accessible renewable energy sources.",
            "Wind energy uses large turbines to convert the kinetic energy from wind into mechanical energy, which is then turned into electricity. Wind farms can be installed both on land and offshore, where wind speeds are higher. Wind energy is a clean, renewable resource that can provide a significant portion of global energy needs.",
            "Hydropower, or hydroelectric power, generates electricity by using the force of flowing water. Dams are built on rivers to create a reservoir, and the water is released through turbines, converting the kinetic energy of the water into electrical power. Hydropower is one of the oldest and most established renewable energy sources.",
            "Geothermal energy comes from the heat stored beneath the Earth's surface. It is accessed through wells drilled deep into the Earth to reach hot water or steam. Geothermal power plants convert this thermal energy into electricity. It's a reliable and consistent energy source that can be used for heating or power generation.",
            "Renewable energy is important because it reduces our dependence on finite fossil fuels, lowers carbon emissions, and helps combat climate change. By transitioning to renewables, we can also create jobs in clean energy industries and promote energy security by diversifying energy sources."
        ]
    },
    "sustainable_lifestyle": {
        "patterns": ["What is a sustainable lifestyle?", "How can I live sustainably?", "Tips for a sustainable life?", "What is sustainable fashion?", "How do I reduce my carbon footprint at home?", "What are the benefits of a sustainable lifestyle?"],
        "responses": [
            "A sustainable lifestyle involves making choices that minimize harm to the environment, conserve resources, and promote social well-being. This can include reducing waste, conserving water, using renewable energy, and supporting companies with environmentally friendly practices. Sustainable living seeks to create a balance between meeting present needs and preserving resources for future generations.",
            "Living sustainably starts with reducing your consumption of single-use items like plastic bags, bottles, and packaging. Opt for reusable products, buy in bulk, and choose eco-friendly alternatives. Be mindful of your energy use by turning off lights, using energy-efficient appliances, and reducing heating and cooling demands.",
            "Sustainable fashion focuses on ethical production, reducing textile waste, and using eco-friendly materials such as organic cotton or recycled fabrics. By buying less, choosing quality over quantity, and supporting brands that prioritize sustainability, you can contribute to reducing the fashion industry's environmental impact.",
            "To reduce your carbon footprint at home, consider upgrading to energy-efficient appliances, using LED lights, and installing insulation to reduce heating and cooling costs. Switching to a plant-based diet, reducing water usage, and recycling can also significantly reduce your environmental impact."
        ]
    },
    "climate_change": {
        "patterns": ["What is climate change?", "Causes of climate change?", "Effects of global warming?", "How to fight climate change?", "What are greenhouse gases?", "What can I do to stop climate change?"],
        "responses": [
            "Climate change refers to long-term changes in temperature, weather patterns, and global ecosystems, largely driven by human activities. The main cause of climate change is the burning of fossil fuels, which releases greenhouse gases like carbon dioxide into the atmosphere, trapping heat and raising global temperatures.",
            "The primary causes of climate change include deforestation, industrial emissions, and the burning of fossil fuels for transportation, energy production, and agriculture. These activities increase the concentration of greenhouse gases in the atmosphere, which leads to global warming and severe weather events.",
            "The effects of global warming include rising sea levels due to melting ice caps, more frequent and intense heatwaves, flooding from storms and heavy rainfall, and disruptions to ecosystems and agriculture. Global warming is also contributing to the loss of biodiversity as many species struggle to adapt to rapid changes in their environments.",
            "To fight climate change, we must reduce greenhouse gas emissions by transitioning to renewable energy, promoting energy efficiency, and adopting sustainable agricultural practices. Individuals can contribute by driving less, reducing energy use, recycling, and supporting policies that encourage climate action.",
            "Greenhouse gases like carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O) trap heat in the Earth's atmosphere, creating the greenhouse effect. This natural process is vital for maintaining life on Earth, but human activities have intensified it, leading to global warming and climate change.",
            "To stop climate change, we must take immediate action to reduce our carbon emissions. This includes transitioning to renewable energy, protecting forests, supporting sustainable agriculture, and reducing waste. You can help by adopting a low-carbon lifestyle, advocating for policies that prioritize climate action, and supporting organizations that work on environmental issues."
        ]
    }
}

# Data preparation
def preprocess_intents(intents):
    patterns = []
    labels = []
    for intent, data in intents.items():
        for pattern in data["patterns"]:
            patterns.append(pattern.lower())  # Convert to lowercase for uniformity
            labels.append(intent)
    return patterns, labels

patterns, labels = preprocess_intents(intents)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)  # Convert patterns to numerical features

# Train the classifier
classifier = LogisticRegression()
classifier.fit(X, labels)  # Fit the data to the classifier

# Generate chatbot response
def get_response(user_input):
    user_input = user_input.lower()  # Lowercase the input
    user_vector = vectorizer.transform([user_input])  # Vectorize user input
    prediction = classifier.predict(user_vector)[0]  # Predict intent
    responses = intents[prediction]["responses"]  # Fetch responses
    return random.choice(responses)  # Randomly select a response

# Combined Streamlit app
st.title("Sustainability Hub 🌱")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose a feature", ["Chatbot", "News", "Research", "Home"])

if option == "Home":
    st.subheader("Welcome to the Sustainability Hub!")
    st.write("Explore news, research, and a chatbot focused on Sustainable Development Goals (SDGs).")
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/UN_SDG_Logo.png", width=300)

elif option == "Chatbot":
    st.subheader("Chatbot for Sustainability")

    # Store conversation in session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display chat history
    for message in st.session_state['messages']:
        st.write(f"**{message['role']}**: {message['content']}")

    # User input
    user_input = st.text_input("Ask a question about sustainability:", "")

    if user_input:
        # Add user message to conversation
        st.session_state['messages'].append({"role": "User", "content": user_input})

        # Get chatbot response and add to conversation
        bot_response = get_response(user_input)
        st.session_state['messages'].append({"role": "Bot", "content": bot_response})

        # Display updated chat history
        st.write(f"**Bot**: {bot_response}")

elif option == "News":
    st.subheader("Latest News on SDGs")
    news_articles = fetch_news()
    for article in news_articles:
        st.write(f"### {article['title']}")
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")

elif option == "Research":
    st.subheader("Research Papers & Wikipedia Summary")
    query = st.text_input("Enter a topic related to SDGs", placeholder="e.g., Climate Change")
    if query:
        st.write("### Wikipedia Summary")
        summary = fetch_wikipedia_summary(query)
        st.write(summary)

        st.write("### Research Papers")
        research_links = fetch_research(query)
        for link in research_links:
            st.markdown(f"- [Research Paper]({link})")

