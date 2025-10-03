from loguru import logger
import os
from dotenv import load_dotenv

# Make sure TTSSpeakFrame is imported
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.runner.run import main
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.runner.types import RunnerArguments
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor



from prompts import  BANK_PROMPT_ARY

from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiVADParams,
    GeminiMultimodalModalities,
    InputParams
)
from pipecat.transcriptions.language import Language
from pipecat.services.gemini_multimodal_live.events import (
    StartSensitivity,
    EndSensitivity
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.transcript_processor import TranscriptProcessor


# select your network protocol
transport_params={
    "webrtc":lambda: TransportParams(
        audio_in_enabled = True,
        audio_out_enabled = True,
         vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    )
}

# Tool Scehma
# my_tools = ToolsSchema(
#     standard_tools=[
#     ]
# )


load_dotenv()
async def run_example(transport: BaseTransport):
    logger.info("Starting bot")

    params = InputParams(
        temperature=0.7,
        modalities=GeminiMultimodalModalities.AUDIO ,
        language=Language.EN_CA,
        vad=GeminiVADParams(
            start_sensitivity=StartSensitivity.HIGH,  # Detect speech quickly
            end_sensitivity=EndSensitivity.LOW,       # Allow longer pauses
            prefix_padding_ms=300,                    # Keep 300ms before speech
            silence_duration_ms=1000,                 # End turn after 1s silence
        )
    )
    llm = GeminiMultimodalLiveLLMService(
        api_key=os.environ["GEMINI_API_KEY"],
        voice_id="Zephyr",
        params=params,
        system_instruction=BANK_PROMPT_ARY,  # Use system instruction instead of user message
        )
    # if you want to add tools:
    # llm.register_function(
    #     "tool_name",
    #     tool_function,
    #     cancel_on_interruption=False,
    # )

    transcript = TranscriptProcessor()

    # Start with an empty context - no initial user message needed
    messages = [{"role": "user", "content": "Say hello!"}]

    context = OpenAILLMContext(messages) #tools=my_tools add it if you have tools
    context_aggregator = llm.create_context_aggregator(context)
    # RTVI: Front Observer
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    # Pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,#
            context_aggregator.user(),  # User responses
            transcript.user(),
            llm,  # LLM
            transport.output(),  # Transport bot output
            transcript.assistant(),
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )
    # Task Pipeline
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True
        ),
        observers=[RTVIObserver(rtvi)]
    )
    
    # Event Handlers
    @transcript.event_handler("on_transcript_update")
    async def handle_update(processor,frame):
        print("*************Messages*************")
        print(frame.messages)

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client Ready")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([context_aggregator.user().get_context_frame()])


    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # Run Pipeline
    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)

async def bot(runner_args: RunnerArguments):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(
        audio_in_enabled = True,
        audio_out_enabled = True,
         vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    )
    await run_example(pipecat_transport)

if __name__ == "__main__":
    main()