from tqdm import tqdm

def train(
    self,
    prob_focus_present: float = 0.0,
    focus_present_mask: Optional[torch.Tensor] = None,
    log_fn: Callable[[dict], None] = noop
):
    """
    Main training loop. Trains the model for a number of steps, updating the model, EMA, and sampling periodically.
    
    Args:
        prob_focus_present: Probability of focusing on the present.
        focus_present_mask: Mask for focusing on specific frames.
        log_fn: Logging function that takes a dictionary as input.
    """
    assert callable(log_fn)

    # Initialize tqdm progress bar for the total number of training steps
    with tqdm(total=self.train_num_steps, desc="Training", unit="step") as pbar:
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                # Get the next batch of data
                data = next(self.dl)

                if len(data) == 2:
                    video_data, text_data = data
                else:
                    video_data = data
                    text_data = None

                video_data = video_data.to(self.device)

                with autocast(enabled=self.amp):  # Automatic mixed precision
                    if text_data is not None:
                        loss = self.model(
                            video_data,
                            cond=text_data,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask
                        )
                    else:
                        loss = self.model(
                            video_data,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask
                        )

                    # Backpropagate loss
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

            # Update tqdm bar with the current loss
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            log = {'loss': loss.item()}

            if exists(self.max_grad_norm):
                # Gradient clipping
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)  # Optimizer step
            self.scaler.update()  # Update the scaler
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_model_every == 0:
                milestone = self.step // self.save_model_every
                self.save(milestone)

            log_fn(log)  # Log the information
            self.step += 1

    print('Training completed.')
