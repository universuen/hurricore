import json
import time
import os
import io
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI


class OpenAIBatchInferencer:
    def __init__(
            self,
            model_name: str = "gemini-3-pro",
            api_key: str | None = os.getenv("OPENAI_API_KEY"),
            base_url: str | None = os.getenv("OPENAI_API_BASE_URL_FOR_BATCH_INFERENCE"),
            use_tqdm: bool = True,
        ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.use_tqdm = use_tqdm
    
    def __call__(
            self, 
            conversations: list[list[dict]],
            max_file_size_mb: int = 50,
            max_workers: int = 16,
            **kwargs,
        ) -> list[str]:
        max_file_size_bytes = max_file_size_mb * 1024 * 1024

        batches_of_lines = []
        batches_of_ids = []
        
        current_batch_lines = []
        current_batch_ids = []
        current_batch_size = 0
        
        print("Preparing batch files...")
        
        for idx, conversation in enumerate(conversations):
            # Construct the request object
            request_item = {
                "custom_id": str(idx), 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": self.model_name, 
                    "messages": conversation, 
                    **kwargs,
                }
            }
            
            # Serialize to get size
            json_line = json.dumps(request_item) + "\n"
            line_size = len(json_line.encode('utf-8'))
            
            # Check if adding this line exceeds the file limit
            # Note: If a single line is larger than max_file_size_bytes, 
            # it will be added to an empty batch effectively ignoring the limit for that specific item.
            if current_batch_lines and (current_batch_size + line_size > max_file_size_bytes):
                # Save current batch
                batches_of_lines.append(current_batch_lines)
                batches_of_ids.append(current_batch_ids)
                
                # Reset
                current_batch_lines = []
                current_batch_ids = []
                current_batch_size = 0
            
            current_batch_lines.append(json_line)
            current_batch_ids.append(str(idx))
            current_batch_size += line_size
            
        # Add the final batch if it exists
        if current_batch_lines:
            batches_of_lines.append(current_batch_lines)
            batches_of_ids.append(current_batch_ids)

        num_batches = len(batches_of_lines)
        print(f"Split into {num_batches} batches based on size limit of {max_file_size_bytes / (1024*1024):.2f}MB.")

        global_results_table = {}
        
        with ThreadPoolExecutor(max_workers=min(max_workers, num_batches)) as executor:
            # Pass the pre-serialized lines and the IDs contained in this batch
            future_to_batch = {
                executor.submit(self.batch_inference, batch_lines, batch_ids): batch_idx
                for batch_idx, (batch_lines, batch_ids) in enumerate(zip(batches_of_lines, batches_of_ids))
            }
            
            iterable = as_completed(future_to_batch)
            if self.use_tqdm:
                iterable = tqdm(iterable, total=num_batches, desc="Processing batches")
                
            for future in iterable:
                batch_idx = future_to_batch[future]
                try:
                    batch_result_dict = future.result()
                    global_results_table.update(batch_result_dict)
                except Exception as e:
                    print(f"Batch {batch_idx} failed with error: {e}")
                    traceback.print_exc()
                    # Fill failed IDs with None
                    failed_ids = batches_of_ids[batch_idx]
                    for fid in failed_ids:
                        global_results_table[fid] = None
        
        final_results = []
        for i in range(len(conversations)):
            final_results.append(global_results_table.get(str(i)))
        
        return final_results


    def batch_inference(
            self, 
            batch_lines: list[str],
            batch_ids: list[str],
        ) -> dict:
        """
        Args:
            batch_lines: List of pre-serialized JSON strings (one per request).
            batch_ids: List of custom_ids (str) contained in this batch.
        Returns:
            dict: Mapping of custom_id -> result content
        """
        
        batch_id = self.create_batch(batch_lines)
        
        try:
            results_table = dict()
            while True:
                batch = self.client.batches.retrieve(batch_id)
                if batch.status in ('completed', 'failed', 'cancelled', 'expired'):
                    break
                time.sleep(1)
            
            if batch.status != 'completed':
                raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}. Error: {batch.errors}")

            if batch.output_file_id:
                file_content = self.download_file(batch.output_file_id).read().decode('utf-8')
                
                for line in file_content.splitlines():
                    result = json.loads(line)
                    custom_id = result['custom_id']
                    try:
                        results_table[custom_id] = result['response']['body']['choices'][0]['message']['content']
                    except BaseException as e:
                        print(f"Failed to process query {custom_id}: {result}")
                        results_table[custom_id] = None
            
            # Ensure all IDs in this batch are in the table (fill missing with None)
            final_batch_results = {}
            for bid in batch_ids:
                final_batch_results[bid] = results_table.get(bid)
                
            return final_batch_results    

        except BaseException as e:
            print(f"Batch inference failed locally. Batch ID: {batch_id}")
            current_status = self.client.batches.retrieve(batch_id).status
            if current_status not in ('completed', 'failed', 'cancelled', 'expired'):
                self.client.batches.cancel(batch_id)
                print(f"Batch inference cancelled remotely. Batch ID: {batch_id}")
            raise e


    def create_batch(
        self, 
        batch_lines: list[str],
    ) -> str:
        # construct_file logic moved here effectively since we already have lines
        buffer = io.BytesIO()
        for line in batch_lines:
            buffer.write(line.encode('utf-8'))
        buffer.seek(0)
        
        file_id = self.upload_file(buffer)
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id


    def upload_file(self, buffer: io.BytesIO) -> str:
        file = self.client.files.create(
            file=("batch_requests.jsonl", buffer),
            purpose="batch"
        )
        return file.id


    def download_file(self, output_file_id: str) -> io.BytesIO:
        return self.client.files.content(output_file_id)


    def get_incompleted_batches(self):
        return [batch for batch in self.client.batches.list(limit=100000).data if batch.status not in ('completed', 'failed', 'cancelled', 'expired')]
    

    def kill_all_batches(self):
        all_batches = self.get_incompleted_batches()
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.client.batches.cancel, batch.id) for batch in all_batches if batch.status not in ('completed', 'failed', 'cancelled', 'expired')]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Killing batches"):
                try:
                    future.result()
                except BaseException as e:
                    print(f"Failed to kill batch: {e}")


if __name__ == "__main__":
    inferencer = OpenAIBatchInferencer(
        api_key="UxaLqO5HgovLaSN08BD2kzpx@3211",
        base_url="http://v2.open.venus.oa.com/llmproxy/v1",
    )
    # Example usage
    results = inferencer(
        [
            # "What is the capital of France?",
            # "What is the capital of Germany?",
            # [Image.new("RGB", (512, 512), color="red"), "What is the color of the image?"]
            [{"role": "user", "content": "What is the capital of France?"}]
        ] * 20,
        max_file_size_mb=1,
    )
    for i in results:
        print(i)
    print(f"Got {len(results)} results")