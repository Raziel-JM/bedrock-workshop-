# README para Generación de Texto y Extracción de Entidades usando Amazon Bedrock

## Descripción General

Este repositorio contiene dos proyectos principales que demuestran el uso de Amazon Bedrock para la generación de texto y la extracción de entidades utilizando modelos avanzados de lenguaje. Estos proyectos destacan las capacidades de las API de Bedrock para generar texto relevante en contexto y extraer información estructurada de datos no estructurados.

## Contenidos

- [00_text_generation_w_bedrock.ipynb](./00_text_generation_w_bedrock.ipynb) - Demuestra el uso del modelo Amazon Titan en Amazon Bedrock para tareas de generación de texto. Incluye un ejemplo de creación de respuestas personalizadas por correo electrónico para retroalimentación de clientes utilizando prompts en escenarios de cero ejemplos (zero-shot). El notebook cubre:
  - Configuración del entorno con las credenciales necesarias de AWS.
  - Configuración de la API de Amazon Bedrock con el cliente `boto3`.
  - Generación de texto con parámetros personalizables como temperatura y top-p.
  - Exploración de generación de salidas en streaming para textos más largos.

- [01_code_generation_w_bedrock.ipynb](./01_code_generation_w_bedrock.ipynb) - Se centra en el uso de Amazon Bedrock para generar fragmentos de código y automatizar tareas repetitivas de programación. Este notebook incluye:
  - Configuración de Amazon Bedrock para casos de uso de generación de código.
  - Ejemplos de generación de funciones en Python basadas en descripciones textuales.
  - Ajuste de parámetros de generación para mayor precisión y relevancia.

- [04_entity_extraction.md](./04_entity_extraction.md) - Explora la extracción de entidades utilizando el modelo Anthropic Claude en Amazon Bedrock. Este notebook demuestra:
  - Extracción de datos estructurados como nombres de libros, nombres de clientes y preguntas a partir de correos electrónicos.
  - Uso de etiquetas XML para una extracción precisa de datos.
  - Personalización de prompts para mejorar la precisión en el reconocimiento de entidades.
  - Análisis y procesamiento de las salidas del modelo utilizando herramientas como `BeautifulSoup`.

---

## Requisitos

- **Python 3.x**
- **Cuenta de AWS** con acceso a Amazon Bedrock.
- Paquetes de Python necesarios:
  - `boto3`
  - `awscli`
  - `langchain-aws`
  - `beautifulsoup4` (para el análisis de XML)

Instala las dependencias con el siguiente comando:

```bash
pip install boto3 awscli langchain-aws beautifulsoup4
```

