from pathlib import Path

out = Path(__file__).parent/'data'/'train_v2.txt'
blocks=[]

def qa(q,a):
    blocks.append(f"User: {q}\nPersona: {a}\n")

# Core identity / behavior
core=[
("Hello!", "Hello! I’m Persona. How can I help?"),
("What are you?", "I’m Persona, a language model trained from random initial weights. I learn patterns by predicting the next token during training."),
("Are you a commercial AI API?", "No. This Persona model is designed to run as its own neural model rather than calling an external AI provider."),
("What if you do not know?", "I should say I’m not sure instead of inventing an answer."),
("Can you learn from corrections?", "Yes. Corrections can be saved as training examples and included in later training runs."),
("How does a transformer learn language?", "During training, the transformer sees token sequences and learns weights that make the correct next token more likely. Repeating this over many examples changes the weights."),
("What is attention?", "Attention lets each position assign different importance to earlier positions when building its representation."),
("What is a token?", "A token is a unit of text processed by a language model. Persona currently uses bytes, so every possible byte value has a direct token id."),
]
for q,a in core: qa(q,a)

# General knowledge, math, science, computing, language
facts = [
("What is 2 + 2?", "2 + 2 = 4."),
("What is 12 times 8?", "12 × 8 = 96."),
("What is 144 divided by 12?", "144 ÷ 12 = 12."),
("What is the square root of 81?", "The principal square root of 81 is 9."),
("What is a fraction?", "A fraction represents a part of a whole or a ratio, such as 3/4."),
("What is a prime number?", "A prime number is an integer greater than 1 with exactly two positive divisors: 1 and itself."),
("What is gravity?", "Gravity is the interaction that causes masses to attract one another. Near Earth, it gives objects a downward acceleration of about 9.8 m/s²."),
("What is photosynthesis?", "Photosynthesis is the process plants and some other organisms use to convert light energy into chemical energy, commonly using carbon dioxide and water to make sugars and release oxygen."),
("What is DNA?", "DNA is a molecule that stores hereditary biological information in living organisms and many viruses."),
("What is a cell?", "A cell is the basic structural and functional unit of life."),
("What is electricity?", "Electricity involves electric charge and its movement or interactions. Electric current is the flow of charge through a material."),
("What is an atom?", "An atom is the basic unit of an element, consisting of a nucleus surrounded by electrons."),
("What is the solar system?", "The solar system is the Sun and the objects gravitationally bound to it, including planets, dwarf planets, moons, asteroids, and comets."),
("What is Earth?", "Earth is the third planet from the Sun and the only astronomical object currently known to support life."),
("What is water made of?", "A water molecule contains two hydrogen atoms and one oxygen atom, written H2O."),
("What is a computer operating system?", "An operating system manages hardware and provides common services and interfaces for applications."),
("What is Linux?", "Linux is a family of operating systems built around the Linux kernel and combined with user-space software and tools."),
("What is an API?", "An API is an interface that lets software components communicate through defined requests, inputs, outputs, and rules."),
("What is HTTP?", "HTTP is a protocol used to transfer resources and messages between clients and servers on the web."),
("What is JSON?", "JSON is a text format for representing structured data using objects, arrays, strings, numbers, booleans, and null."),
("What is Python?", "Python is a general-purpose programming language known for readable syntax and a large ecosystem of libraries."),
("What is JavaScript?", "JavaScript is a programming language widely used for interactive web applications and also used on servers and other platforms."),
("What is HTML?", "HTML is the markup language used to structure content on web pages."),
("What is CSS?", "CSS is a stylesheet language used to control the presentation and layout of HTML and other documents."),
("What is a database?", "A database is an organized system for storing and retrieving structured or semi-structured information."),
("What is encryption?", "Encryption transforms information using a key so that only authorized parties can recover the intended plaintext."),
("What is authentication?", "Authentication verifies who a user or service is. Authorization determines what that authenticated identity is allowed to do."),
("What is a neural network?", "A neural network is a parameterized function made from layers of mathematical operations whose parameters can be adjusted during training."),
("What is machine learning?", "Machine learning is a method of building systems whose behavior is learned from data rather than being completely hand-coded."),
("What is overfitting?", "Overfitting happens when a model learns training examples too specifically and performs poorly on new examples."),
("What is a gradient?", "A gradient is a collection of derivatives that points toward the direction of greatest local increase of a function."),
("Why use train and validation data?", "Training data is used to update model weights. Validation data is kept separate so we can estimate how well the model generalizes."),
]
for q,a in facts: qa(q,a)

# Instruction patterns
patterns = [
("Explain a difficult idea simply.", "Start with the main idea in plain language, then give a small example, and only add technical detail if it helps."),
("Give me a short answer.", "Give the direct answer first and avoid unnecessary background."),
("I made a mistake. What should you do?", "Acknowledge the correction, update the answer, and avoid repeating the known mistake."),
("How should you handle uncertainty?", "Separate what is known from what is uncertain. If current or precise information is required, say that it should be checked."),
("How should you answer a yes or no question?", "Answer yes or no first, then add one short explanation when useful."),
("How should you solve a math problem?", "Identify the given values, choose the appropriate rule or equation, calculate carefully, and check the result."),
("How should you debug a program?", "Reproduce the problem, inspect the error and relevant inputs, isolate the smallest failing part, change one thing at a time, and test again."),
("How should you write code?", "Prefer clear names, small functions, simple control flow, useful error handling, and tests for important behavior."),
]
for q,a in patterns: qa(q,a)

# Reusable mini dialogues
for topic, turns in {
"weather": [("What should I do if the weather is very hot?", "Stay hydrated, avoid prolonged direct sun, and choose a cool place during the hottest part of the day."),
             ("Should you pretend to know tomorrow's weather?", "No. Weather forecasts change and should be checked with a current weather source.")],
"coding": [("My program crashes. What should I send you?", "Send the error message, the smallest relevant code section, what you expected, and what actually happened."),
            ("Should I change ten things at once while debugging?", "Usually no. Change one relevant thing at a time so you can tell which change affected the result.")],
"learning": [("How can I learn programming?", "Build small projects, practice regularly, read documentation, debug your own mistakes, and gradually increase project complexity."),
              ("Is memorizing everything necessary?", "No. Understanding concepts and knowing how to find reliable documentation are more useful than memorizing every detail.")],
"conversation": [("Can you keep an answer concise?", "Yes. I can give the direct answer first and expand only when needed."),
                  ("Can you ask for clarification?", "Yes, when the missing information would materially change the correct answer.")]
}.items():
    for q,a in turns: qa(q,a)

# Generate many varied question forms from the factual bank so the model sees conversational structure.
for i,(q,a) in enumerate(facts):
    subject=q.rstrip('?')
    variants=[
        f"Tell me about {subject.lower().replace('what is ','')}",
        f"Can you explain {subject.lower().replace('what is ','')}",
        f"What do you know about {subject.lower().replace('what is ','')}",
    ]
    for v in variants:
        qa(v, a)

# Small coding examples in natural language.
code_examples=[
("How do I print Hello in Python?", "Use print(\"Hello\")."),
("How do I define a Python function?", "Use def name(parameters): followed by an indented function body."),
("How do I create a JavaScript constant?", "Use const name = value; when the binding should not be reassigned."),
("How do I select an HTML element with CSS by id?", "Use #id, for example #menu { ... }."),
("What does a loop do?", "A loop repeats a block of instructions while a condition holds or for each item in a sequence."),
("What does an if statement do?", "An if statement chooses whether to execute code based on a condition."),
]
for q,a in code_examples: qa(q,a)

# Repeat selected examples with slight conversational context to teach turn-taking.
for n in range(12):
    qa(f"Round {n+1}: what should Persona remember about answering?", "Be direct, be honest, explain clearly, and never claim to know something that was not established.")

out.write_text("\n".join(blocks), encoding='utf-8')
print(f"wrote {out} ({len(out.read_text(encoding='utf-8')):,} chars, {len(blocks)} examples)")
