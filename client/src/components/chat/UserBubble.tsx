import type { GenerateRequest } from '../../api/types';

export function UserBubble({ req }: { req: GenerateRequest }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-md bg-accent text-on-accent px-4 py-2.5 shadow-hard">
        <p className="leading-snug">{req.prompt}</p>
        <p className="mt-1 font-mono text-[10px] opacity-75">
          {req.student_method} · seed {req.seed} · {req.steps} steps
        </p>
      </div>
    </div>
  );
}
