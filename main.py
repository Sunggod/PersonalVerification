import hashlib
import os
import io
import json
import time
import base64
import numpy as np
from PIL import ImageGrab, Image
import easyocr
import torch
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import re

class PersonalTrainerAdvancedAnalyzer:
    def __init__(self, historico_path='profile_history.json'):
        # Configuração de logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename='personal_trainer_analyzer.log'
        )
        
        # Configurações de OCR
        self.text_reader = easyocr.Reader(['pt', 'en'])
        
        # Configuração do modelo de IA generativa
        genai.configure(api_key="AIzaSyBtEmWrAO_8Q9I-zlMxHtoV3X1a9ONyx9g")
        
        self.generation_config = {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
        )
        
        # Histórico de análises
        self.historico_path = historico_path
        self.historico = self._carregar_historico()
        
        # Vetorização para similaridade
        self.vectorizer = TfidfVectorizer()
        
        # Palavras-chave de personal trainer com maior prioridade
        self.personal_trainer_keywords = [
            'personal trainer', 'educador físico', 'professor de educação física', 
            'treinador', 'preparador físico', 'coach fitness', 'instrutor fitness',
            'profissional de educação física', 'cref', 'treinamento', 'fitness'
        ]
        
        # Palavras-chave indicando formação profissional
        self.professional_training_keywords = [
            'cref', 'registro profissional', 'graduação', 'licenciatura', 
            'bacharelado', 'pós-graduação', 'especialização', 'mestrado', 
            'doutorado', 'universidade', 'faculdade', 'educação física'
        ]
        
        # Lista expandida de palavras-chave de exclusão com mais nuances
        self.excluded_professions_keywords = [
            # Área da saúde
            'fisioterapeuta', 'fisioterapia', 'nutricionista', 'nutrição', 
            'médico', 'médica', 'enfermeiro', 'enfermeira', 'psicólogo', 'psicóloga',
            'terapeuta', 'fonoaudiólogo', 'fonoaudióloga', 'nutricionista esportivo',
            
            # Outros campos que podem ser confundidos
            'professor', 'professor universitário', 'pesquisador', 'cientista',
            'biomédico', 'nutricionista clínico', 'nutricionista esportivo',
            
            # Termos de áreas correlatas mas não específicas de treinamento
            'fisiologia', 'biomecânica', 'reabilitação', 'hospital', 'clínica',
            
            # Termos técnicos que podem gerar confusão
            'prescrição de exercício', 'avaliação física', 'saúde', 'bem-estar'
        ]
        
        # Nova lista de prioridade de personal trainer
        self.personal_trainer_priority_keywords = [
            'personal trainer', 'treinador', 'instrutor fitness', 
            'preparador físico', 'coach fitness', 'educador físico', 
            'treinamento', 'fitness', 'cref'
        ]
    
    def _carregar_historico(self):
        """Carregar histórico de análises"""
        try:
            if os.path.exists(self.historico_path):
                with open(self.historico_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {'analises': []}
        except Exception as e:
            logging.error(f"Erro ao carregar histórico: {e}")
            return {'analises': []}
    
    def _salvar_historico(self, nova_analise):
        """Salvar nova análise no histórico"""
        try:
            self.historico['analises'].append(nova_analise)
            
            # Limitar tamanho do histórico
            if len(self.historico['analises']) > 1000:
                self.historico['analises'] = self.historico['analises'][-1000:]
            
            with open(self.historico_path, 'w', encoding='utf-8') as f:
                json.dump(self.historico, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Erro ao salvar histórico: {e}")
    
    def _verificar_exclusao_profissional(self, texto):
        """
        Verificação inteligente de exclusão profissional
        Considera contexto, prioridade de personal trainer e formação em Ed. Física
        """
        if not texto:
            return False

        texto_lower = texto.lower()

        # Nova verificação de formação em Educação Física
        formacao_educacao_fisica = any(
            keyword in texto_lower 
            for keyword in ['educação física', 'd.física', 'bacharel em educação física', 'graduação em ed. física', 'ed física']
        )

        # Se tiver formação em Educação Física, não excluir
        if formacao_educacao_fisica:
            logging.info(f"Formação em Educação Física detectada. Ignorando exclusão.")
            return False

        # Verificar se há menção de personal trainer na primeira etapa
        tem_personal_trainer = any(
            keyword in texto_lower 
            for keyword in self.personal_trainer_priority_keywords
        )

        # Contar ocorrências de palavras de exclusão
        exclusoes = [
            keyword for keyword in self.excluded_professions_keywords 
            if keyword in texto_lower
        ]

        # Lógica de decisão inteligente
        if tem_personal_trainer:
            # Se personal trainer for mencionado, priorizar
            if len(exclusoes) <= 1:  # Tolerância para 1 menção de profissão
                return False  # Não excluir
            else:
                logging.warning(f"Múltiplas profissões detectadas: {exclusoes}")
                return False  # Ainda assim, não excluir totalmente

        if exclusoes:
            # Se não tem personal trainer, mas tem outras profissões
            logging.info(f"Profissões encontradas: {exclusoes}")
            return True

        return False

    
    def _calcular_similaridade_historico(self, nova_analise):
        """Calcular similaridade com análises históricas"""
        try:
            if not self.historico['analises']:
                return 0
            
            textos_historico = [analise.get('texto_extraido', '') for analise in self.historico['analises']]
            textos_historico.append(nova_analise.get('texto_extraido', ''))
            
            # Vetorizar textos
            vetores = self.vectorizer.fit_transform(textos_historico)
            
            # Calcular similaridade do último vetor (nova análise) com histórico
            similaridades = cosine_similarity(vetores[-1:], vetores[:-1])[0]
            
            return max(similaridades) if len(similaridades) > 0 else 0
        except Exception as e:
            logging.error(f"Erro ao calcular similaridade: {e}")
            return 0
    
    def _verificar_formacao_profissional(self, texto):
        """
        Verificar se o texto indica formação profissional em educação física
        Agora com verificação de exclusão
        """
        texto_lower = texto.lower()
        
        # Verificar exclusões primeiro
        if self._verificar_exclusao_profissional(texto):
            return False
        
        # Se não for excluído, verificar formação
        for keyword in self.professional_training_keywords:
            if keyword in texto_lower:
                return True
        return False
    
    def _parsear_resposta_ia(self, resposta_ia_texto):
        """
        Parsear a resposta da IA para extrair informações-chave
        """
        try:
            # Usar regex para extrair informações-chave
            classificacao_match = re.search(r'Classificação: (.*)', resposta_ia_texto)
            probabilidade_match = re.search(r'Probabilidade: (\d+)%', resposta_ia_texto)
            tipo_perfil_match = re.search(r'Tipo de Perfil: (.*)', resposta_ia_texto)
            
            return {
                'classificacao': classificacao_match.group(1) if classificacao_match else None,
                'probabilidade': int(probabilidade_match.group(1)) if probabilidade_match else 0,
                'tipo_perfil': tipo_perfil_match.group(1) if tipo_perfil_match else None
            }
        except Exception as e:
            logging.error(f"Erro ao parsear resposta da IA: {e}")
            return None
    
    def _analisar_perfil_privado(self, texto, resposta_ia_texto):
        """
        Analisar perfil com informações flexíveis, considerando formação em Ed. Física
        mesmo com outras profissões principais
        """
        texto_lower = texto.lower()

        # Verificar menções a formação em Educação Física primeiro
        formacao_educacao_fisica = any(
            keyword in texto_lower 
            for keyword in ['educação física', 'Ed.física', 'bacharel em educação física', 'graduação em ed. física','Ed. Física', 'ed.fisica']
        )

        # Verificar exclusões considerando formação em Ed. Física
        if self._verificar_exclusao_profissional(texto):
            # Se tiver formação em Ed. Física, permite passar pela exclusão
            if formacao_educacao_fisica:
                logging.info(f"Perfil tem outra profissão, mas possui formação em Educação Física. Liberando análise.")
            else:
                return {    
                    'eh_personal_trainer': False,
                    'probabilidade': 0,
                    'justificativa': "Profissão excluída explicitamente"
                }
        
        # Parsear resposta da IA
        analise_ia = self._parsear_resposta_ia(resposta_ia_texto)
        
        if not analise_ia:
            # Fallback para método original
            eh_personal_trainer = (
                any(keyword in texto_lower for keyword in self.personal_trainer_keywords) or 
                formacao_educacao_fisica
            )
            
            probabilidade = 0
            justificativa = "Perfil com informações limitadas"
            
            if eh_personal_trainer and formacao_educacao_fisica:
                probabilidade = 90
                justificativa = "Formação em Educação Física confirmada"
            elif eh_personal_trainer:
                probabilidade = 60
                justificativa = "Menções indiretas a treinamento"
            elif formacao_educacao_fisica:
                probabilidade = 80
                justificativa = "Formação em Educação Física identificada"
            
            return {
                'eh_personal_trainer': eh_personal_trainer,
                'probabilidade': probabilidade,
                'justificativa': justificativa
            }
        
        # Usar análise da IA considerando formação
        eh_personal_trainer = (
            (analise_ia['classificacao'] == 'PERSONAL TRAINER PROFISSIONAL' and 
            analise_ia['probabilidade'] > 50) or 
            formacao_educacao_fisica
        )
        
        return {
            'eh_personal_trainer': eh_personal_trainer,
            'probabilidade': max(analise_ia['probabilidade'], 80 if formacao_educacao_fisica else 0),
            'justificativa': f"Análise IA: {analise_ia['classificacao']} (Probabilidade: {analise_ia['probabilidade']}%). Formação em Ed. Física: {formacao_educacao_fisica}"
        }
    
    def extrair_texto_easyocr(self, imagem):
        """Extração de texto com fallback para baixa informação"""
        try:
            imagem_np = np.array(imagem)
            resultados = self.text_reader.readtext(imagem_np)
            
            # Processar resultados com confiança
            texto_extraido = []
            for (bbox, text, conf) in resultados:
                # Considerar apenas texto com confiança alta
                if conf > 0.5:
                    texto_extraido.append(text)
            
            texto_final = " ".join(texto_extraido).strip()
            
            # Fallback para texto muito curto
            if len(texto_final) < 10:
                logging.warning("Texto extraído muito curto, possível perfil com baixa informação")
            
            return texto_final
        
        except Exception as e:
            logging.error(f"Erro na extração de texto: {e}")
            return ""
    
    def processar_imagem(self, imagem):
        """Processamento avançado da imagem com priorização da análise IA e filtros de exclusão flexíveis"""
        try:
            # Análise de texto
            texto_extraido = self.extrair_texto_easyocr(imagem)
            
            # Nova verificação de formação em Educação Física antes da exclusão
            texto_lower = texto_extraido.lower()
            formacao_educacao_fisica = any(
            keyword in texto_lower 
            for keyword in ['educação física', 'Ed.física', 'bacharel em educação física', 'graduação em ed. física','Ed. Física', 'ed.fisica']
            )
            
            # Verificação de exclusão considerando formação em Educação Física
            if self._verificar_exclusao_profissional(texto_extraido) and not formacao_educacao_fisica:
                return {
                    'texto_extraido': texto_extraido,
                    'eh_personal_trainer': False,
                    'probabilidade': 0,
                    'justificativa': "Profissão excluída após análise detalhada",
                    'timestamp': time.time()
                }
            
            # Converter imagem para base64 para análise IA
            buffered = io.BytesIO()
            imagem.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Prompt de análise avançada com instruções de exclusão
            prompt = f"""Análise Profunda de Perfil de Personal Trainer:
            
            Contexto do Texto Extraído: {texto_extraido}
            
            INSTRUÇÕES CRÍTICAS DE EXCLUSÃO:
            - NÃO classifique como Personal Trainer se encontrar:
            * Fisioterapeutas
            * Nutricionistas
            * Médicos
            * Enfermeiros
            * Psicólogos
            * Profissionais de saúde
            
            Instruções para Análise:
            1. Identifique se o perfil é de um PERSONAL TRAINER PROFISSIONAL
            2. Considere FORMAÇÃO TÉCNICA em EDUCAÇÃO FÍSICA
            3. Diferencie entre profissionais e entusiastas
            4. EXCLUA RIGOROSAMENTE profissionais de outras áreas
            5. Seja preciso e objetivo
            6. Se tiver menções a ed.fisica podemos considerar como personal trainer!
            
            Formato da Resposta:
            ```
            Classificação: [PERSONAL TRAINER PROFISSIONAL/NÃO PROFISSIONAL]
            Probabilidade: [0-100%]
            Tipo de Perfil: [Profissional/Atleta/Entusiasta]
            Justificativa Detalhada: [Motivos precisos]
            Insights Adicionais: [Observações complementares]
            ```"""
            
            # Análise IA com instruções de exclusão
            resposta_ia = self.model.generate_content([
                {'mime_type': 'image/png', 'data': img_base64},
                prompt
            ])
            
            # Análise de perfil usando resultado da IA
            if len(texto_extraido) < 30:
                analise_perfil = self._analisar_perfil_privado(texto_extraido, resposta_ia.text)
            else:
                # Para perfis com mais informações, usar análise da IA diretamente
                analise_ia = self._parsear_resposta_ia(resposta_ia.text)
                
                analise_perfil = {
                    'eh_personal_trainer': (
                        analise_ia['classificacao'] == 'PERSONAL TRAINER PROFISSIONAL' and 
                        analise_ia['probabilidade'] > 50
                    ),
                    'probabilidade': analise_ia['probabilidade'],
                    'justificativa': resposta_ia.text
                }
            
            # Criar objeto de análise final
            analise = {
                'texto_extraido': texto_extraido,
                'analise_ia': resposta_ia.text,
                'eh_personal_trainer': analise_perfil['eh_personal_trainer'] or formacao_educacao_fisica,
                'probabilidade': max(analise_perfil.get('probabilidade', 0), 80 if formacao_educacao_fisica else 0),
                'justificativa': (
                    analise_perfil.get('justificativa', 'Sem justificativa') + 
                    (". Formação em Educação Física detectada" if formacao_educacao_fisica else "")
                ),
                'timestamp': time.time()
            }
            
            # Calcular similaridade com histórico
            similaridade = self._calcular_similaridade_historico(analise)
            analise['similaridade_historico'] = similaridade
            
            # Salvar no histórico
            self._salvar_historico(analise)
            
            return analise
        
        except Exception as e:
            logging.error(f"Erro no processamento de imagem: {e}")
            return None

def main():
    print("[INFO] Iniciando Sistema de Análise Avançada de Personal Trainers")
    analyzer = PersonalTrainerAdvancedAnalyzer()
    
    # Método para gerar hash de imagem
    def get_image_hash(image):
        """Gerar hash para detectar imagens duplicadas"""
        if not image:
            return None
        
        # Redimensionar imagem para um tamanho consistente para hash
        resized_image = image.resize((100, 100), Image.LANCZOS)
        # Converter para escala de cinza para hash mais consistente
        grayscale_image = resized_image.convert('L')
        
        # Gerar hash
        hash_object = hashlib.md5(np.array(grayscale_image).tobytes())
        return hash_object.hexdigest()
    
    # Configuração para monitoramento
    CHECK_INTERVAL = 5  # segundos entre verificações de imagem
    LAST_IMAGE_HASH = None
    
    while True:
        try:
            imagem = ImageGrab.grabclipboard()
            
            if isinstance(imagem, Image.Image):
                # Gerar hash para imagem atual
                current_image_hash = get_image_hash(imagem)
                
                # Verificar se a imagem é diferente da última analisada
                if current_image_hash != LAST_IMAGE_HASH:
                    # Processar imagem com análise avançada
                    resultado = analyzer.processar_imagem(imagem)
                    
                    if resultado:
                        print("\n--- Resultado da Análise ---")
                        print(f"Texto Extraído: {resultado.get('texto_extraido', 'N/A')}")
                        print(f"É Personal Trainer: {resultado.get('eh_personal_trainer', False)}")
                        
                        # Imprimir análise IA se existir
                        if 'analise_ia' in resultado:
                            print(f"Análise IA:\n{resultado.get('analise_ia', 'Sem análise')}")
                        
                        # Imprimir probabilidade e justificativa para perfis privados
                        if 'probabilidade' in resultado:
                            print(f"Probabilidade: {resultado.get('probabilidade', 0)}%")
                            print(f"Justificativa: {resultado.get('justificativa', 'N/A')}")
                        
                        print(f"Similaridade com Histórico: {resultado.get('similaridade_historico', 0):.2%}")
                    
                    # Atualizar hash da última imagem
                    LAST_IMAGE_HASH = current_image_hash
                else:
                    print("[INFO] Imagem duplicada. Ignorando análise.")
                
            # Aguardar entre verificações
            time.sleep(CHECK_INTERVAL)
        
        except Exception as e:
            logging.error(f"Erro no monitoramento: {e}")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()