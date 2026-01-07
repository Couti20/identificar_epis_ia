# 🛡️ Sistema de Detecção de EPIs v3.0

Sistema de visão computacional para monitoramento de Equipamentos de Proteção Individual (EPIs) em tempo real usando YOLO.

## ✨ Novidade: Zoom de Atenção (AI Focus)

Quando a IA detecta um EPI com **dúvida** (confiança entre 50% e 75%), ela automaticamente:
1. **Recorta** a região de interesse
2. **Amplia** (zoom digital 2x)
3. **Reanalisa** com mais detalhes
4. **Confirma** ou descarta a detecção

Ideal para objetos pequenos como **óculos**, **luvas** e **protetor auricular**.

## 📋 Requisitos

- Python 3.8+
- Webcam (integrada ou USB)
- Modelo YOLO treinado (`oitenta-tres.pt`)

## 📦 Instalação

```bash
pip install opencv-python ultralytics numpy
```

## 🚀 Execução

```bash
python main.py
```

## ⚙️ Configuração

Edite a classe `Config` em `main.py`:

### Configurações Básicas
| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `MODELO_PATH` | Caminho do modelo YOLO | `'oitenta-tres.pt'` |
| `CONFIANCA` | Confiança mínima (0-1) | `0.40` |
| `CAMERA_INDEX` | Índice da câmera | `0` |
| `IOU_LIMITE` | IoU mínimo para validar EPI | `0.40` |

### Configurações de Zoom de Atenção
| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `ZOOM_ATIVADO` | Liga/desliga o zoom | `True` |
| `ZOOM_CONFIANCA_MINIMA` | Abaixo disso = muito incerto | `0.50` |
| `ZOOM_CONFIANCA_MAXIMA` | Acima disso = certo | `0.75` |
| `ZOOM_FATOR` | Fator de ampliação | `2.0` |
| `ZOOM_CLASSES_PRIORITARIAS` | Classes que recebem zoom | `[8, 9, 2, 14]` |

## 🎯 EPIs Detectados

| EPI | Parte do Corpo | Alerta | Zoom? |
|-----|----------------|--------|-------|
| Capacete | Cabeça | SEM CAPACETE | ❌ |
| Colete | Corpo | SEM COLETE | ❌ |
| Óculos | Rosto | SEM ÓCULOS | ✅ |
| Luvas | Mãos | SEM LUVAS | ✅ |
| Botas | Pés | SEM BOTAS | ✅ |
| Protetor Auricular | Orelha | - | ✅ |

## 🎨 Cores

- 🟢 **Verde**: EPI detectado com certeza
- 🔴 **Vermelho**: EPI ausente (não-conformidade)
- 🔵 **Ciano**: EPI confirmado por ZOOM
- 🟡 **Amarelo**: Informações neutras

## ⌨️ Controles

- `Q` - Encerrar aplicação

## 📁 Estrutura do Projeto

```
ia_camera/
├── main.py              # Aplicação principal (com Zoom de Atenção)
├── oitenta-tres.pt      # Modelo YOLO (83% precisão)
├── requirements.txt     # Dependências
└── README.md            # Este arquivo
```

## 🔬 Como Funciona o Zoom de Atenção

```
┌─────────────────────────────────────────────────────────┐
│  Frame Original (1920x1080)                             │
│                                                         │
│    ┌──────┐                                             │
│    │ 🔍  │  Detecção com 60% de confiança              │
│    │Óculos│  (está na "faixa de dúvida")               │
│    └──────┘                                             │
│         ↓                                               │
│    ┌──────────────┐                                     │
│    │              │  Região recortada + margem         │
│    │   ZOOM 2x    │  Ampliada para 640x640             │
│    │              │  Reanalizada pelo modelo           │
│    └──────────────┘                                     │
│         ↓                                               │
│    Resultado: 85% de confiança → CONFIRMADO ✓          │
└─────────────────────────────────────────────────────────┘
```

## 📝 Licença

Projeto interno para uso em ambientes industriais.
