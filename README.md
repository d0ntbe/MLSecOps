MLSecOps

Список инструментов, ресурсов и руководств по MLSecOps с открытым исходным кодом (Machine Learning Security Operations).
	
Методика выражена с помощью четырех таблиц, где:

Таблица 1 — открытые инструменты для аудита безопасности ML (OpenSource);
Таблица 2 — коммерческие решения для аудита безопасности ML;
Таблица 3 — материалы и практики для построения процесса MLSecOps;
Таблица 4 — список общих регуляторных источников и мировых практик, которые могут быть использованы для построения процесса MLSecOps.

Таблица 1 - Инструменты для аудита ML

| | | |
| ------------- | ------------- | ------------- | 
| Advbox  | Набор инструментов для генерации состязательных примеров, которые обманывают нейронные сети  PaddlePaddle, PyTorch, Caffe2, MXNet, Keras, TensorFlow.  | https://github.com/advboxes/AdvBox |
| Adversarial_ml_ctf  | Уязвимые образы систем (docker) с ML для проведения CTF  | https://github.com/arturmiller/  |
| Advertorch  | Инструментарий Python для исследования состязательной устойчивости. В частности, AdverTorch содержит модули для генерации состязательных возмущений и защиты от состязательных примеров, а также скрипты для обучения состязательности.  | https://github.com/BorealisAI/advertorch |
| Advmlthreatmatrix  | Фреймворк по аналогии с Atlas | https://github.com/mitre/advmlthreatmatrix |
| Ai-exploits  | Коллекция эксплойтов AI/ML  | https://github.com/protectai/ai-exploits |
| Ai-goat  | Уязвимые образы систем (docker) с ML для проведения CTF  | https://github.com/dhammon/ai-goat |
| AnonLLM  | Реализация методов обезличивания данных в ML   | https://github.com/fsndzomga/anonLLM |
| ART(Adversarial-robustness-toolbox)  | Библиотека на Python для обеспечения безопасности машинного обучения. ART предоставляет инструменты, которые позволяют разработчикам и исследователям защищать и оценивать модели и приложения машинного обучения от таких враждебных угроз, как уклонение, отравление, извлечение информации и другие. ART поддерживает многие популярные платформы машинного обучения (TensorFlow, Keras, PyTorch, MXNet, scikit-learn, XGBoost, LightGBM, CatBoost, GPy и т.д.) и многие типы данных (изображения, таблицы, аудио, видео и т.д.). Позволяет создавать, например, Adversarial-контент для проверки качества работы ML-моделей и оценки качества. | https://github.com/Trusted-AI/adversarial-robustness-toolbox |
| ARX  | Инструмент для анонимизации наборов данных  | https://arx.deidentifier.org/ |
| AugLy  | Инструмент для генерации состязательных атак | https://github.com/facebookresearch/AugLy |
| Awesome-MLSecOps  | Набор полезных материалов и ссылок по MLSecOps (один из лучших)  | https://github.com/RiccardoBiosas/awesome-MLSecOps |
| CleverHans  | Библиотеки Python для тестирования уязвимостей систем машинного обучения к Adversarial атакам.  | https://github.com/cleverhans-lab/cleverhans |
| DamnVulnerableLLMProject  | Уязвимый образ системы (docker) с ML для тренировок по SecOps  | https://github.com/harishsg993010/ |
| Deep-pwning  | Фреймворк для экспериментов с моделями машинного обучения с целью оценки их надежности. Инструмент Pentest’a.  | https://github.com/cchio/deep-pwning |
| Differential-privacy-library  | Оценка приватности моделей  | https://github.com/IBM/differential-privacy-library  |
| Fml-security  | Практические примеры уязвимость ML в рамках SSDL  | https://github.com/EthicalML/fml-security  |
| Foolbox  | Набор инструментов Python для создания и оценки состязательных атак и защит от них.  | https://github.com/bethgelab/foolbox |
| Garak  | Cканер уязвимостей LLM. Проверка на галюцинации, утечку данных, prompt injection, misinformation, toxicity generation, jailbreaks и другие типы уязвимостей ML  | https://github.com/leondz/garak |
| Giskard  | Сканер для обнаружения Hallucinations, Harmful content generation, Prompt injection, Robustness issues, Sensitive information disclosure, Stereotypes & discrimination и так далее.  | https://github.com/Giskard-AI/giskard |
| Knockoffnets  | Реализации атак BlackBox с целью кражи данных модели.  | https://github.com/tribhuvanesh/knockoffnets |
| LintML  | Сканер (подсвечивает) для выявления Plaintext credentials, Unsafe deserialization, Serialization to unsafe formats, Using untrustworthy assets.  | https://github.com/JosephTLucas/lintML |
| ML_security_study_map  | Карта для изучения ML  | https://github.com/wearetyomsmnv/AI-LLM-ML_security_study_map |
| MLSploit  | Это облачный фреймворк для интерактивных экспериментов с исследованиями в области состязательного машинного обучения. Работает через REST API.  | https://github.com/mlsploit/ |
| Model-Inversion-Attack-ToolBox  | Инструмент проведения атак инверсии на модели.  | https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox |
| Models-are-code  | Описания и исследования векторов атак на модели  | https://hiddenlayer.com/research/models-are-code/  |
| ModelScan  | Сканер для выявления возможных атак сериализации ML-моделей.  | https://github.com/protectai/modelscan  |
| NB Defense  | Программа NB Defense, разработанная компанией Protect AI, представляет собой расширение для JupyterLab (IDE для создания моделей), а также инструмент командной строки, который на каждом этапе разработкип подсветит небезопасный код.   | https://nbdefense.ai/ |
| NeMo Guardrails  | Добавление защиты при разработке от атак на conversational applications.  | https://github.com/NVIDIA/NeMo-Guardrails |
| OpenAttack  | Инструментарий для проведения текстовой состязательной атаки на основе Python с открытым исходным кодом, который обрабатывает весь процесс текстовой состязательной атаки, включая предварительную обработку текста, доступ к модели жертвы, создание примеров состязательной атаки и оценку.  | https://github.com/thunlp/OpenAttack |
| Pallms  | Набор полезных нагрузок для атак на ML  | https://github.com/mik0w/pallms |
| Privacy Meter  | Библиотека с открытым исходным кодом для аудита конфиденциальности данных в статистических алгоритмах и алгоритмах машинного обучения.  | https://github.com/privacytrustlab/ml_privacy_meter |
| PromptInject  | Фреймворк, который собирает prompts по модульному принципу, чтобы обеспечить количественный анализ устойчивости LLMS к  prompt-атакам.  | https://github.com/agencyenterprise/PromptInject  |
| PyRIT | Инструмент для оценки рисков в рамках использования ИИ | https://github.com/Azure/PyRIT |
| Raze_to_the_ground_aisec23 | Инструмент для выявления фишинга на страницах HTML, основанный на ML | https://github.com/advmlphish/raze_to_the_ground_aisec23 |
| Safetensors | Обеспечивает безопасную работу с тензорами (представление входных данных (например, изображения, тексты) в виде числовых массивов) | https://github.com/huggingface/safetensors |
| Stealing_DL_Models | Генерирует копию сверточной нейронной сети, запрашивая ее как черный ящик со случайными данными для создания помеченных областей для предсказания. Позволяет steal моделей.  | https://github.com/jeiks/Stealing_DL_Models |
| Tensorflow  Model-analysis | Библиотека для анализа, валидации и мониторинга моделей машинного обучения в промышленной эксплутации. | https://github.com/tensorflow/model-analysis |
| TensorFlow Privacy | Библиотека Python, которая включает в себя реализации оптимизаторов TensorFlow для обучения моделей машинного обучения с дифференциальной конфиденциальностью. Библиотека поставляется с учебными пособиями, а также инструментами анализа и расчета конфиденциальности. | https://github.com/tensorflow/privacy |
| TextAttack | Сканер на языке Python для проведения состязательных атак. | https://github.com/QData/TextAttack |
| TextFooler | Модель для атак на системы обработки естественного языка (NLP), которая использует специально созданные тексты для обмана моделей. Эти атаки могут быть направлены на изменение классификации текста, нарушение работы систем понимания текста или даже на генерацию вредоносных текстов| https://github.com/jind11/TextFooler |
| Vger | Инструмент постэксплуатации для ML, преимущественно направленный на Jupyter. | https://github.com/JosephTLucas/vger |
| Vigil-llm | Библиотека на Python и REST API сканер для анализа запросов и ответов на запросы Large Language Model для обнаружения уязвимостей. Этот репозиторий содержит сигнатуры обнаружения и наборы данных, необходимые для анализа. | https://github.com/deadbits/vigil-llm |
| Watchtower | Открытый инструмент для сканирования моделей, встраивается в CI/CD. | https://github.com/bosch-aisecurity-aishield/watchtower |
| Аudit-ai | Тестирование на предвзятость, ошибки прогнозов   для целей принятия социально значимых решений. Может строить графики bias (отклонение от ожидаемого) | https://github.com/pymetrics/audit-ai |
| | | |

Таблица 2 — коммерческие решения для аудита безопасности ML;

| | | |
| ------------- | ------------- | ------------- | 
| Сitadel | Сканер безопасности моделей и данных для обучения. Встраивается в CI/CD. Включает статический анализ. | https://www.citadel.co.jp/en/products/lens/ |
| AI Validation | Сканер безопасности моделей. Встраивается в CI/CD. | https://www.robustintelligence.com/platform/ai-validation |
| Guardian | Защита моделей в CI/CD при разработке. Включает статический анализ. | https://protectai.com/guardian |
| Veil | Инструмент для анонимизации наборов данных. | https://veil.ai/ |
| Guard-ai | Облачное решение для оценки безопасности моделей | https://github.com/guard-ai |
| АI-guard | Анонимайзер на базе ИИ  | https://springlabs.com/ai-guard/|
| | | |

Таблица 3 - Материалы и рекомендации по MLSecOps, которые могут быть использованы для построения процесса безопасности ML.

| | | |
| ------------- | ------------- | ------------- | 
| Ai-security-101 | Лучшая на 2024 год методология и описание процессов по защите/атак AI  | https://www.nightfall.ai/ai-security-101 |
| Atlas MITRE | Рекомендации от MITRE (табличка угроз, методология) для AI  | https://atlas.mitre.org/ |
| LLM-attacks | Статья от PortSwigger об атаках на ML, если используется Web | https://portswigger.net/web-security/llm-attacks |
| AI-red-team | Рекомендации от  Microsoft для RedTeam в рамках ML | https://learn.microsoft.com/en-us/security/ai-red-team/ |
| AI-risk-assessment | Рекомендации от  Microsoft для управления рисками для ML | https://learn.microsoft.com/en-us/security/ai-red-team/ai-risk-assessment |
| OWASP | Классификация Top-10 уязвимостей от OWASP для ML | https://llmtop10.com/|
| Assessment-list (EU) | Оценочный лист для самооценки в рамках цифровой безопасности EU | https://digital-strategy.ec.europa.eu/en/library/assessment-list-trustworthy-artificial-intelligence-altai-self-assessment |
| | | |

Таблица 4 - Список общих регуляторных источников и мировых практик, которые могут быть использованы для построения процесса MLSecOps

| | | |
| ------------- | ------------- | ------------- | 
| ASVS (Application Security Verification Standard)| Cтандарт безопасности приложений, разработанный OWASP. | https://owasp.org/ |
| BSIMM (Building Security In Maturity Mode) |  Построение зрелой модели обеспечения безопасности | https://bsimm.com/ |
| CIS-CONTROLS (Center for Internet Security Controls) | Практики по обеспечению безопасности информационных систем. | https://www.cisecurity.org/ |
| CWE | Каталог известных уязвимостей | https://cwe.mitre.org/ |
| CycloneDX-SBOM | SBOM от OWASP | https://cyclonedx.org/|
| ENISA (Агентство Европейского союза по кибербезопасности) | Открытая документация | https://www.enisa.europa.eu/ |
| ISO27001 | Международный стандарт по информационной безопасности. | https://www.iso.org/ |
| MITREATT&CK | База знаний о тактиках, техниках и процедурах, используемых злоумышленниками. | https://attack.mitre.org/ |
| NIST | Национальный институт стандартов и технологий. | https://www.nist.gov/ |
| OPENCRE | Открытые общие требования от OWASP | https://github.com/OWASP/OpenCRE |
| SAMM | Software Assurance Maturity Model | https://www.opensamm.org/ |
| | | |

















